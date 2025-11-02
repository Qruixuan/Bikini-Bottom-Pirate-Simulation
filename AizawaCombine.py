"""
AizawaCombine.py
把 MerchantShip.py + pirate.py + NavyAgent.py 三个文件合成一个版本
并在模拟结束后用 matplotlib 画出所有 agent 的轨迹。

依赖:
    pip install mesa matplotlib
"""

import math
import random
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

import matplotlib
# 强制 Matplotlib 使用 'TkAgg' 后端，这是最标准的 GUI 后端之一
matplotlib.use('TkAgg')


# ============================================================
# 通用工具
# ============================================================
def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """计算两点之间的欧几里得距离."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def step_move(pos: Tuple[float, float],
              target: Tuple[float, float],
              dist: float) -> Tuple[float, float]:
    """从 pos 朝 target 走 dist 这么远（连续坐标）"""
    x, y = pos
    tx, ty = target
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy) or 1e-9
    if dist >= d:
        return (tx, ty)
    return (x + dist * dx / d, y + dist * dy / d)


# ============================================================
# 商船（用你 MerchantShip.py 的逻辑）:contentReference[oaicite:3]{index=3}
# ============================================================
class MerchantAgent(Agent):
    """一个有固定航线、动态警戒和两档速度的商船."""
    STATE_SAILING = "sailing"
    STATE_EVADING = "evading"
    STATE_IN_PORT = "in_port"

    def __init__(
            self, unique_id, model,
            route: List[Tuple[float, float]],
            normal_speed_kn=12,
            evasion_speed_kn=18,
            visibility_nm=8,
            alert_param=0.05,
            port_wait_time_hrs=5.0
    ):
        super().__init__(unique_id, model)
        self.route = route
        self.normal_speed = normal_speed_kn
        self.evasion_speed = evasion_speed_kn
        self.visibility = visibility_nm
        self.alert_param = alert_param
        self.port_wait_time = port_wait_time_hrs

        # 状态管理
        self.state = self.STATE_SAILING
        self.current_route_index = 0
        self.wait_timer = 0.0
        self.awareness = False
        self.last_known_pirate_pos = None

        # 给海军看的标志（来自 NavyAgent 模式）:contentReference[oaicite:4]{index=4}
        self.under_attack = False

    # --- 辅助方法 ---

    def _get_current_target(self) -> Tuple[float, float]:
        """获取航线上的下一个目标点."""
        if not self.route or self.current_route_index >= len(self.route):
            return self.pos
        return self.route[self.current_route_index]

    def _move_towards(self, dest, speed_kn, hours):
        """
        移动逻辑（已修正边界限制和移除检查）。
        此方法将钳制新位置，确保它不会超出 ContinuousSpace 的边界。
        """
        if self.pos is None:
            return False

        step = speed_kn * hours
        cur = self.pos
        dx, dy = dest[0] - cur[0], dest[1] - cur[1]
        d = math.hypot(dx, dy)

        arrived = False
        if d <= step or d == 0:
            new_pos = dest
            arrived = True
        else:
            new_pos = (cur[0] + dx / d * step, cur[1] + dy / d * step)

        # 边界
        x_max = getattr(self.model.space, 'x_max', self.model.space.width)
        y_max = getattr(self.model.space, 'y_max', self.model.space.height)
        clamped_x = max(0.0, min(new_pos[0], x_max))
        clamped_y = max(0.0, min(new_pos[1], y_max))
        final_pos = (clamped_x, clamped_y)

        if final_pos != new_pos:
            arrived = False

        self.model.space.move_agent(self, final_pos)
        return arrived

    def receive_distress(self, pirate_pos):
        """
        商船发现/遭遇海盗时触发，标记自己 under_attack，
        并通知所有海军（如果模型里有）:contentReference[oaicite:5]{index=5}
        """
        self.under_attack = True
        self.last_known_pirate_pos = pirate_pos
        if hasattr(self.model, "navy_agents"):
            for n in self.model.navy_agents:
                n.receive_distress(self)

    def step(self):
        hours = self.model.hours_per_step

        # 1. 检查附近海盗并更新警戒
        pirate_threat = self._check_pirate_threat()

        if self.state == self.STATE_SAILING:
            if pirate_threat:
                self.state = self.STATE_EVADING
                self.last_known_pirate_pos = pirate_threat
            else:
                self._sail_route(hours)

        elif self.state == self.STATE_EVADING:
            if pirate_threat:
                self.last_known_pirate_pos = pirate_threat
                self._evade(hours)
            else:
                # 没威胁了，恢复航行
                self.state = self.STATE_SAILING
                self.last_known_pirate_pos = None
                self.under_attack = False
                self._sail_route(hours)

        elif self.state == self.STATE_IN_PORT:
            self._wait_in_port(hours)

    # --- 内部行为方法 ---

    def _check_pirate_threat(self):
        closest_pirate_pos = None
        min_distance = float('inf')

        for agent in self.model.schedule.agents:
            if agent.__class__.__name__ == 'PirateAgent':
                if agent.pos is None:
                    continue
                d = distance(self.pos, agent.pos)

                if d <= self.visibility:
                    alert_threshold = 1.0 - d * self.alert_param
                    alert_threshold = max(0.0, min(1.0, alert_threshold))

                    if random.random() < alert_threshold:
                        if d < min_distance:
                            min_distance = d
                            closest_pirate_pos = agent.pos
                            self.awareness = True

        return closest_pirate_pos

    def _sail_route(self, hours):
        target = self._get_current_target()
        arrived = self._move_towards(target, self.normal_speed, hours)

        if arrived:
            self.current_route_index += 1
            if self.current_route_index >= len(self.route):
                if self.route:
                    self.state = self.STATE_IN_PORT
                else:
                    self.state = self.STATE_SAILING

    def _evade(self, hours):
        if self.last_known_pirate_pos is None or self.pos is None:
            self.state = self.STATE_SAILING
            self.under_attack = False
            return

        pirate_pos = self.last_known_pirate_pos
        dx, dy = self.pos[0] - pirate_pos[0], self.pos[1] - pirate_pos[1]

        x_max = getattr(self.model.space, 'x_max', self.model.space.width)
        y_max = getattr(self.model.space, 'y_max', self.model.space.height)
        escape_distance = x_max + y_max

        d = math.hypot(dx, dy)
        if d == 0:
            return

        escape_dest = (self.pos[0] + dx / d * escape_distance, self.pos[1] + dy / d * escape_distance)

        self._move_towards(escape_dest, self.evasion_speed, hours)

        # 逃跑时继续呼救
        self.receive_distress(pirate_pos)

    def _wait_in_port(self, hours):
        self.wait_timer += hours
        if self.wait_timer >= self.port_wait_time:
            self.current_route_index = 0
            self.state = self.STATE_SAILING
            self.under_attack = False


# ============================================================
# 海盗（用你 pirate.py 的复杂 FSM 版）:contentReference[oaicite:6]{index=6}
# ============================================================
class PirateAgent(Agent):
    """Pirate behavior cycle: select → cruise → search → pursuit/attack → recuperate → return"""
    STATE_SELECT = "select"
    STATE_CRUISE = "cruise"
    STATE_SEARCH = "search"
    STATE_PURSUIT = "pursuit"
    STATE_ATTACK = "attack"
    STATE_RECUP = "recuperate"
    STATE_RETURN = "return_home"

    def __init__(
            self, unique_id, model,
            home_anchor=(0, 0),
            cruising_speed_kn=10,
            pursuit_speed_kn=28,
            endurance_days=14,
            visibility_nm=80,
            attack_time_hrs=0.5,
            cool_down_hrs=2,
            navy_knowledge_prob=0.4,
            qa=0.2, qu=0.5
    ):
        super().__init__(unique_id, model)
        self.home_anchor = home_anchor
        self.cruising_speed = cruising_speed_kn
        self.pursuit_speed = pursuit_speed_kn
        self.endurance = endurance_days * 24.0
        self.visibility = visibility_nm
        self.attack_time = attack_time_hrs
        self.cool_down = cool_down_hrs
        self.navy_knowledge = navy_knowledge_prob
        self.qa = qa
        self.qu = qu

        self.state = PirateAgent.STATE_SELECT
        self.time_since_departure = 0.0
        self.target_cell = None
        self.search_time = 0.0
        self.current_target_merchant = None
        self.cooldown_timer = 0.0

    # --- Main step ---
    def step(self):
        hours = self.model.hours_per_step
        if self.state == self.STATE_SELECT:
            self._select_target_area()
        elif self.state == self.STATE_CRUISE:
            self._cruise(hours)
        elif self.state == self.STATE_SEARCH:
            self._search(hours)
        elif self.state == self.STATE_PURSUIT:
            self._pursue(hours)
        elif self.state == self.STATE_ATTACK:
            self._attack(hours)
        elif self.state == self.STATE_RECUP:
            self._recuperate(hours)
        elif self.state == self.STATE_RETURN:
            self._return_home(hours)

    # --- Internal behaviors ---
    def _select_target_area(self):
        # 原版是看 model.merchant_density_grid，这里可能没有，就改成随机挑一块
        grid = getattr(self.model, "merchant_density_grid", None)
        if grid and len(grid) > 0:
            merged = {}
            for cell_pos, val in grid.items():
                weight = val
                if (random.random() < self.navy_knowledge) and hasattr(self.model, "navy_positions"):
                    for npos in self.model.navy_positions:
                        d = distance(cell_pos, npos)
                        if d < 200:
                            weight *= 0.5
                merged[cell_pos] = max(weight, 0.0)
            total = sum(merged.values())
            if total <= 0:
                self.target_cell = random.choice(list(grid.keys()))
            else:
                r, cum = random.random() * total, 0.0
                for pos, val in merged.items():
                    cum += val
                    if r <= cum:
                        self.target_cell = pos
                        break
        else:
            x = random.uniform(0, self.model.space.x_max)
            y = random.uniform(0, self.model.space.y_max)
            self.target_cell = (x, y)
        self.state = self.STATE_CRUISE

    def _move_towards(self, dest, speed_kn, hours):
        if self.pos is None or dest is None:
            return
        step = speed_kn * hours
        cur = self.pos
        dx, dy = dest[0] - cur[0], dest[1] - cur[1]
        d = math.hypot(dx, dy)

        if d <= step or d == 0:
            new_pos = dest
        else:
            new_pos = (cur[0] + dx / d * step, cur[1] + dy / d * step)

        x_max = getattr(self.model.space, 'x_max', self.model.space.width)
        y_max = getattr(self.model.space, 'y_max', self.model.space.height)
        clamped_x = max(0.0, min(new_pos[0], x_max))
        clamped_y = max(0.0, min(new_pos[1], y_max))
        final_pos = (clamped_x, clamped_y)

        self.model.space.move_agent(self, final_pos)

    def _cruise(self, hours):
        if self.target_cell is None:
            self.state = self.STATE_SELECT
            return
        self._move_towards(self.target_cell, self.cruising_speed, hours)
        if distance(self.pos, self.target_cell) < 1.0:
            self.state = self.STATE_SEARCH
            self.search_time = 0.0

    def _search(self, hours):
        self.search_time += hours
        cur = self.pos
        jitter_x, jitter_y = random.uniform(-1, 1), random.uniform(-1, 1)
        new_pos = (
            max(0, min(self.model.space.x_max, cur[0] + jitter_x)),
            max(0, min(self.model.space.y_max, cur[1] + jitter_y)),
        )
        self.model.space.move_agent(self, new_pos)

        # 找商船
        for agent in self.model.merchant_agents:
            if agent.pos is None:
                continue
            if distance(self.pos, agent.pos) <= self.visibility:
                self.current_target_merchant = agent
                self.state = self.STATE_PURSUIT
                return

        self.time_since_departure += hours
        if self.time_since_departure >= self.endurance:
            self.state = self.STATE_RETURN

    def _pursue(self, hours):
        if self.current_target_merchant is None:
            self.state = self.STATE_SEARCH
            return

        merchant = self.current_target_merchant

        if merchant.state == MerchantAgent.STATE_IN_PORT:
            self.state = self.STATE_SEARCH
            self.current_target_merchant = None
            return

        merchant_pos = merchant.pos
        if merchant_pos is None:
            self.state = self.STATE_SEARCH
            self.current_target_merchant = None
            return

        # 看有没有海军在附近 → 有就跑
        nearest_navy = None
        nearest_navy_dist = float("inf")
        for agent in self.model.navy_agents:
            if agent.pos is None:
                continue
            d = distance(self.pos, agent.pos)
            if d < self.visibility and d < nearest_navy_dist:
                nearest_navy_dist = d
                nearest_navy = agent

        if nearest_navy:
            # 被海军吓跑
            self.current_target_merchant = None
            self.state = self.STATE_RETURN
            return

        # 没有海军 → 继续追
        self._move_towards(merchant_pos, self.pursuit_speed, hours)

        if distance(self.pos, merchant_pos) <= 0.2:
            # 到了，准备开抢
            if merchant.awareness or merchant.state == MerchantAgent.STATE_EVADING:
                merchant.awareness = True
                merchant.receive_distress(self.pos)
            self.state = self.STATE_ATTACK
            self.attack_timer = 0.0

    def _attack(self, hours):
        self.attack_timer += hours
        if self.attack_timer >= self.attack_time:
            merchant = self.current_target_merchant

            # 被 model 删掉了就算了
            if merchant not in self.model.schedule.agents:
                self.state = self.STATE_RECUP
                self.cooldown_timer = 0.0
                self.current_target_merchant = None
                return

            # 劫持概率跟商船速度有关
            s = merchant.normal_speed
            m_base = 10.0
            pa = max(0.0, (2.0 - s / m_base) * self.qa)
            pu = max(0.0, (2.0 - s / m_base) * self.qu)

            prob = pa if merchant.awareness else pu

            if random.random() < prob:
                # 劫持成功
                self.model.hijack_count += 1
                try:
                    self.model.schedule.remove(merchant)
                    merchant.pos = None
                    self.model.merchant_agents = [m for m in self.model.merchant_agents if m is not merchant]
                except Exception:
                    pass

            # 无论成功失败 → 进入恢复
            self.state = self.STATE_RECUP
            self.cooldown_timer = 0.0
            self.current_target_merchant = None

    def _recuperate(self, hours):
        self.cooldown_timer += hours
        if self.cooldown_timer >= self.cool_down:
            self.current_target_merchant = None
            if self.time_since_departure >= self.endurance:
                self.state = self.STATE_RETURN
            else:
                self.state = self.STATE_SEARCH

    def _return_home(self, hours):
        self._move_towards(self.home_anchor, self.cruising_speed, hours)
        if distance(self.pos, self.home_anchor) < 1.0:
            self.time_since_departure = 0.0
            self.state = self.STATE_SELECT


# ============================================================
# 海军（基本保持 NavyAgent.py 的写法）:contentReference[oaicite:7]{index=7}
# ============================================================
class NavyAgent(Agent):
    """
    连续空间海军：
    - 听到商船呼救马上出发
    - 抵达后把商船的 under_attack 清掉
    - 回基地途中也能被新的 distress 打断
    """
    def __init__(self, unique_id, model,
                 speed: float = 40.0,
                 armament: float = 1.0,
                 base_pos: Tuple[float, float] | None = None):
        super().__init__(unique_id, model)
        self.speed = speed
        self.armament = armament
        self.intercept_radius = 1.0
        self.pos_f = (0.0, 0.0)
        self.target = None
        if base_pos is None:
            base_pos = (0.0, self.model.space.y_max)
        self.base_pos = base_pos
        self.state = "idle"

    def receive_distress(self, merchant: MerchantAgent):
        if merchant is None:
            return
        if not getattr(merchant, "under_attack", False):
            return

        if self.target is None:
            self.target = merchant
            self.state = "to_target"
            return

        # 有已有目标 → 选离自己近的那个
        try:
            if distance(self.pos_f, merchant.pos) < distance(self.pos_f, self.target.pos):
                self.target = merchant
                self.state = "to_target"
        except Exception:
            pass

    def step(self):
        hours = self.model.hours_per_step
        dist_per_step = self.speed * hours

        # 主动找一下有没有被打的
        if self.state != "to_target" and self.target is None:
            ua = [m for m in self.model.merchant_agents if getattr(m, "under_attack", False)]
            if ua:
                self.target = min(ua, key=lambda m: distance(self.pos_f, m.pos))
                self.state = "to_target"

        # 去救人
        if self.state == "to_target" and self.target is not None:
            if not getattr(self.target, "under_attack", False):
                self.target = None
                self.state = "rtb"
                self._move_to_base(dist_per_step)
                return

            target_pos = getattr(self.target, "pos", None)
            if target_pos is None:
                self.target = None
                self.state = "rtb"
                self._move_to_base(dist_per_step)
                return

            self.pos_f = step_move(self.pos_f, target_pos, dist_per_step)
            self.model.space.move_agent(self, self.pos_f)

            if distance(self.pos_f, target_pos) <= self.intercept_radius:
                # 拦截成功
                self.target.under_attack = False
                if hasattr(self.model, "events"):
                    self.model.events.append(("DISRUPT", self.unique_id, self.target.unique_id))
                self.target = None
                self.state = "rtb"
            return

        # 回基地
        if self.state == "rtb":
            self._move_to_base(dist_per_step)
            if distance(self.pos_f, self.base_pos) < 0.5:
                self.state = "idle"
            return

        # idle
        if self.state == "idle":
            self.model.space.move_agent(self, self.pos_f)

    def _move_to_base(self, dist):
        self.pos_f = step_move(self.pos_f, self.base_pos, dist)
        self.model.space.move_agent(self, self.pos_f)


# ============================================================
# 模型：把三种 agent 融合在一起:contentReference[oaicite:8]{index=8}
# 并且这里加“轨迹记录”功能
# ============================================================
class NavalSimModel(Model):
    def __init__(self,
                 width=300, height=200,
                 num_pirates=3,
                 num_merchants=6,
                 num_navy=1,
                 hours_per_step=1/6):
        super().__init__()
        self.space = ContinuousSpace(width, height, torus=False)
        self.space.x_max = width
        self.space.y_max = height

        self.schedule = RandomActivation(self)
        self.hours_per_step = hours_per_step

        self.events: list[tuple] = []
        self.hijack_count = 0

        # 给别的 agent 快速访问
        self.merchant_agents: list[MerchantAgent] = []
        self.navy_agents: list[NavyAgent] = []
        self.navy_positions: list[Tuple[float, float]] = []

        # 轨迹记录：id -> list of (x,y)
        self.trajectories: Dict[str, List[Tuple[float, float]]] = {}

        # -------- 航线还是原来的 --------
        port_A = (20, 20)
        port_B = (width - 20, height - 20)
        port_C = (width // 2, height // 2)
        routes = [
            [port_A, port_B, port_A],
            [port_B, port_C, port_A, port_B],
            [port_A, port_C, port_B],
        ]

        # 1) 商船（不动）
        for i in range(num_merchants):
            route = random.choice(routes)
            start_pos = route[0]
            m = MerchantAgent(
                unique_id=f"merchant_{i}",
                model=self,
                route=route,
                normal_speed_kn=random.uniform(10, 14),
                evasion_speed_kn=random.uniform(16, 20),
                visibility_nm=15,
                alert_param=0.03
            )
            self.space.place_agent(m, start_pos)
            self.schedule.add(m)
            self.merchant_agents.append(m)
            self.trajectories[m.unique_id] = [start_pos]

        # 2) 海盗 —— 按你说的范围生成：x 150~300, y 0~75
        for i in range(num_pirates):
            home_x = random.uniform(150, width)       # width=300 → 150~300
            home_y = random.uniform(0, 75)            # 0~75
            home = (home_x, home_y)
            p = PirateAgent(f"pirate_{i}", self, home_anchor=home,
                            visibility_nm=60)
            self.space.place_agent(p, home)
            self.schedule.add(p)
            self.trajectories[p.unique_id] = [home]

        # 3) 海军 —— 固定在 (100, 150)
        for i in range(num_navy):
            base = (100, 150)
            n = NavyAgent(f"navy_{i}", self, base_pos=base)
            n.pos_f = base
            self.space.place_agent(n, base)
            self.schedule.add(n)
            self.navy_agents.append(n)
            self.trajectories[n.unique_id] = [base]

        # 存航线，画图用
        self.routes_template = routes

    def step(self):
        # 更新海军位置给海盗看
        self.navy_positions = [n.pos for n in self.navy_agents]

        self.schedule.step()

        # 记录轨迹
        for agent in list(self.schedule.agents):
            if agent.pos is None:
                continue
            if agent.unique_id not in self.trajectories:
                self.trajectories[agent.unique_id] = []
            self.trajectories[agent.unique_id].append(agent.pos)


    def step(self):
        # 更新海军位置给海盗避让用
        self.navy_positions = [n.pos for n in self.navy_agents]

        self.schedule.step()

        # 记录轨迹
        for agent in list(self.schedule.agents):
            if agent.pos is None:
                continue
            if agent.unique_id not in self.trajectories:
                self.trajectories[agent.unique_id] = []
            self.trajectories[agent.unique_id].append(agent.pos)


# ============================================================
# 模拟 + 画图
# ============================================================
def run_and_plot(steps=200):
    model = NavalSimModel(num_pirates=3, num_merchants=6, num_navy=1,
                          width=300, height=200,
                          hours_per_step=1/6)

    for t in range(steps):
        model.step()

    print("Hijacks:", model.hijack_count)
    print("Events:", model.events)

    # 开始画
    fig, ax = plt.subplots(figsize=(10, 6))

    # 先画航线模版（淡淡的）
    for route in model.routes_template:
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, linestyle='--', color='lightgreen', linewidth=1, alpha=0.5)

    # 再画实际轨迹
    for agent_id, traj in model.trajectories.items():
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]

        if agent_id.startswith("merchant_"):
            ax.plot(xs, ys, color='green', linewidth=1.5, label='Merchant' if 'Merchant' not in ax.get_legend_handles_labels()[1] else "")
            ax.scatter(xs[0], ys[0], color='green', s=20)
        elif agent_id.startswith("pirate_"):
            ax.plot(xs, ys, color='red', linewidth=1.0, label='Pirate' if 'Pirate' not in ax.get_legend_handles_labels()[1] else "")
            ax.scatter(xs[0], ys[0], color='red', s=20)
        elif agent_id.startswith("navy_"):
            ax.plot(xs, ys, color='blue', linewidth=2.0, label='Navy' if 'Navy' not in ax.get_legend_handles_labels()[1] else "")
            ax.scatter(xs[0], ys[0], color='blue', s=30, marker='s')

    ax.set_xlim(0, model.space.x_max)
    ax.set_ylim(0, model.space.y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Maritime ABM Trajectories (Merchants / Pirates / Navy)")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_and_plot(steps=250)
