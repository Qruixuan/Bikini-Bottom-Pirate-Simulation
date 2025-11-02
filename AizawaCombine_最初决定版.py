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
import numpy as np

import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
import matplotlib.animation as animation

EPS = 1e-3
ani = None

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
        self.under_attack = False  # [新增] 攻击状态旗帜
        self.last_known_pirate_pos = None

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
        # 调度同步安全检查：如果 Agent 已被移除，则 self.pos 为 None
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

        # --- 边界钳制/限制 ---
        x_max = getattr(self.model.space, 'x_max', 1000)
        y_max = getattr(self.model.space, 'y_max', 1000)

        # 钳制新位置，确保它不会超出 [0, max] 范围
        clamped_x = max(0.0, min(new_pos[0], x_max))
        clamped_y = max(0.0, min(new_pos[1], y_max))

        final_pos = (clamped_x, clamped_y)
        # ----------------------

        # 如果船被边界限制停住，则视为未到达目标
        if final_pos != new_pos:
            arrived = False

        self.model.space.move_agent(self, final_pos)
        return arrived

    def receive_distress(self, pirate_pos):
        """响应求救信号，供海盗调用. (保留此方法，尽管功能已由 _send_distress_call 替代)"""
        pass

    def step(self):
        hours = self.model.hours_per_step

        # 调度安全检查
        if self not in self.model.schedule.agents:
            return

        # 1. 检查附近海盗并更新警戒
        pirate_threat = self._check_pirate_threat()

        if self.state == self.STATE_SAILING:
            if pirate_threat:
                self.state = self.STATE_EVADING
                self.under_attack = True
                self.last_known_pirate_pos = pirate_threat
                self._send_distress_call()  # [新增] 发送求救信号
            else:
                self._sail_route(hours)

        elif self.state == self.STATE_EVADING:
            if pirate_threat:
                self.last_known_pirate_pos = pirate_threat
                self._evade(hours)
                self._send_distress_call()  # [新增] 持续发送求救信号
            else:
                # 威胁解除，切换回航行状态
                self.state = self.STATE_SAILING
                self.under_attack = False
                self.last_known_pirate_pos = None

                # --- 【智能返航：选择最近的剩余航点作为新目标】 ---
                remaining_route = self.route[self.current_route_index:]
                if remaining_route:
                    # 找到当前位置离哪个剩余航点最近
                    closest_index = min(
                        range(len(remaining_route)),
                        key=lambda i: distance(self.pos, remaining_route[i])
                    )
                    # 将下一个目标点更新为这个最近点
                    self.current_route_index += closest_index
                # ----------------------------------------------------

                self._sail_route(hours)

        elif self.state == self.STATE_IN_PORT:
            self._wait_in_port(hours)
            self.under_attack = False

    # --- 内部行为方法 ---

    def _check_pirate_threat(self):
        # ... (代码不变) ...
        closest_pirate_pos = None
        min_distance = float('inf')

        for agent in self.model.schedule.agents:
            if agent.__class__.__name__ == 'PirateAgent':
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
            self.under_attack = False  # 确保清除状态
            return

        pirate_pos = self.last_known_pirate_pos
        dx, dy = self.pos[0] - pirate_pos[0], self.pos[1] - pirate_pos[1]

        x_max = getattr(self.model.space, 'x_max', 1000)
        y_max = getattr(self.model.space, 'y_max', 1000)
        escape_distance = x_max + y_max

        d = math.hypot(dx, dy)
        if d == 0: return

        # 计算远离海盗的逃跑目的地
        escape_dest = (self.pos[0] + dx / d * escape_distance, self.pos[1] + dy / d * escape_distance)

        self._move_towards(escape_dest, self.evasion_speed, hours)

    def _send_distress_call(self):
        """扫描军舰，并向最近的军舰发送求救信号."""
        closest_navy = None
        min_dist = float('inf')

        for agent in self.model.schedule.agents:
            # 使用类名字符串 'Navy' 进行判断
            if agent.__class__.__name__ == 'Navy':
                # 军舰的连续位置存储在 pos_f 中 (根据您的 Navy Agent 逻辑)
                navy_pos = getattr(agent, "pos_f", None)
                if navy_pos:
                    d = distance(self.pos, navy_pos)
                    if d < min_dist and agent.can_accept_mission(self.pos):
                        min_dist = d
                        closest_navy = agent

        # 如果找到军舰，发送求救信号
        if closest_navy:
            # 军舰的 receive_distress 需要传入商船自身对象
            closest_navy.receive_distress(self)

    def _wait_in_port(self, hours):
        self.wait_timer += hours
        if self.wait_timer >= self.port_wait_time:
            self.current_route_index = 0
            self.state = self.STATE_SAILING

# ============================================================
# 海盗（用你 pirate.py 的复杂 FSM 版）:contentReference[oaicite:6]{index=6}
# ============================================================
class PirateAgent(Agent):
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
            max_sailing_steps=100,
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
        self.max_sailing_steps = max_sailing_steps
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
        self.sailing_steps = 0

    def step(self):
        hours = self.model.hours_per_step

        # ① 先检查警戒区（港口、海军基地）
        if hasattr(self.model, "guard_zones") and self.pos is not None:
            for gz in self.model.guard_zones:
                if distance(self.pos, gz["center"]) <= gz["radius"]:
                    # 记录一个事件可选
                    if hasattr(self.model, "events"):
                        self.model.events.append((
                            "PIRATE_BOUNCED",
                            self.unique_id,
                            gz["label"],
                            float(self.pos[0]),
                            float(self.pos[1]),
                        ))
                    # 触发你已有的回港逻辑
                    self._trigger_return(reason=f"enter_guard_zone:{gz['label']}")
                    return

        # 状态机
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

    def _trigger_return(self, reason=None):
        self.current_target_merchant = None
        self.target_cell = self.home_anchor
        self.state = self.STATE_RETURN
        self.sailing_steps = 0
        if reason:
            print(f"→ Pirate {self.unique_id} triggered return due to: {reason}")

    def _select_target_area(self):
        anchor = getattr(self, "home_anchor", None)
        home_sigma = getattr(self, "home_sigma", 200.0)
        grid = getattr(self.model, "merchant_density_grid", None)

        # 海军基地和港口列表
        navy_bases = [n.base_pos for n in getattr(self.model, "navy_agents", []) if hasattr(n, "base_pos")]
        # 把港口拆成单独点
        port_positions = self.model.ports if hasattr(self.model, "ports") else []

        if grid and len(grid) > 0:
            merged = {}
            for cell_pos, val in grid.items():
                weight = float(val)

                # 1. 偏向 home_anchor
                if anchor is not None:
                    dx = cell_pos[0] - anchor[0]
                    dy = cell_pos[1] - anchor[1]
                    d = math.hypot(dx, dy)
                    sigma = max(1e-6, float(home_sigma))
                    home_factor = math.exp(- (d * d) / (2.0 * sigma * sigma))
                    weight *= home_factor

                # 2. 避开海军基地和港口（距离越近，权重越低）
                avoid_factor = 1.0
                for nb in navy_bases + port_positions:
                    d_nb = math.hypot(cell_pos[0] - nb[0], cell_pos[1] - nb[1])
                    avoid_factor *= math.exp(- (50.0 / (d_nb + 1e-6)) ** 2)  # 50可调节影响范围
                weight *= avoid_factor

                merged[cell_pos] = max(weight, 0.0)

            total = sum(merged.values())
            if total <= 0:
                if anchor is not None:
                    nearest = min(grid.keys(), key=lambda p: math.hypot(p[0]-anchor[0], p[1]-anchor[1]))
                    self.target_cell = nearest
                else:
                    self.target_cell = random.choice(list(grid.keys()))
            else:
                r, cum = random.random() * total, 0.0
                for pos, val in merged.items():
                    cum += val
                    if r <= cum:
                        self.target_cell = pos
                        break
        else:
            # fallback 随机生成
            if anchor is not None:
                sigma = max(1e-6, float(home_sigma))
                x = random.gauss(anchor[0], sigma)
                y = random.gauss(anchor[1], sigma)
                self.target_cell = (x, y)
            else:
                x = random.uniform(0, self.model.space.x_max)
                y = random.uniform(0, self.model.space.y_max)
                self.target_cell = (x, y)

        self.sailing_steps = 0
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
        x_max = getattr(self.model.space, 'x_max', 1000)
        y_max = getattr(self.model.space, 'y_max', 1000)
        clamped_x = max(0.0, min(new_pos[0], x_max - EPS))
        clamped_y = max(0.0, min(new_pos[1], y_max - EPS))
        self.model.space.move_agent(self, (clamped_x, clamped_y))

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
            max(0.0, min(cur[0] + jitter_x, self.model.space.x_max - EPS)),
            max(0.0, min(cur[1] + jitter_y, self.model.space.y_max - EPS)),
        )
        self.model.space.move_agent(self, new_pos)
        for agent in self.model.schedule.agents:
            if isinstance(agent, MerchantAgent):
                if distance(self.pos, agent.pos) <= self.visibility:
                    self.current_target_merchant = agent
                    self.state = self.STATE_PURSUIT
                    return
        self.time_since_departure += hours
        if self.time_since_departure >= self.max_sailing_steps:
            self.state = self.STATE_RETURN

    def _pursue(self, hours):
        if self.current_target_merchant is None:
            self.state = self.STATE_SEARCH
            return
        merchant = self.current_target_merchant
        if merchant.state == MerchantAgent.STATE_IN_PORT or merchant.pos is None:
            self.state = self.STATE_SEARCH
            self.current_target_merchant = None
            return
        self._move_towards(merchant.pos, self.pursuit_speed, hours)
        if distance(self.pos, merchant.pos) <= 0.2:
            if merchant.awareness or merchant.state == MerchantAgent.STATE_EVADING:
                merchant.awareness = True
                merchant.receive_distress(self.pos)
            self.state = self.STATE_ATTACK
            self.attack_timer = 0.0

    def _attack(self, hours):
        """攻击行为，若此时发现海军则立即撤退"""
        # 海军检查（仅在攻击阶段触发）
        if hasattr(self.model, "schedule") and self.pos is not None:
            for agent in self.model.schedule.agents:
                if agent.__class__.__name__ == "NavyAgent" and agent.pos is not None:
                    dnavy = distance(self.pos, agent.pos)
                    if dnavy < 0.5*self.visibility:
                        print(f"⚓ Pirate {self.unique_id} spotted Navy during attack! Retreating!")
                        self._trigger_return(reason="navy_during_attack")
                        return

        self.attack_timer += hours
        if self.attack_timer >= self.attack_time:
            merchant = self.current_target_merchant

            # 商船可能已经被别的逻辑删掉了
            if merchant not in self.model.schedule.agents:
                self.state = self.STATE_RECUP
                self.cooldown_timer = 0.0
                self.current_target_merchant = None
                return

            # ---- 计算成功概率 ----
            s = merchant.normal_speed
            m_base = 10.0
            pa = max(0.0, (2.0 - s / m_base) * self.qa)
            pu = max(0.0, (2.0 - s / m_base) * self.qu)
            prob = pa if merchant.awareness else pu

            if random.random() < prob:
                # ✅ 劫持成功
                self.model.hijack_count += 1

                # ✅ 事件记录
                if hasattr(self.model, "events"):
                    self.model.events.append((
                        "HIJACK",
                        self.unique_id,  # 哪个海盗
                        merchant.unique_id,  # 劫了谁
                        float(self.pos[0]),
                        float(self.pos[1]),
                    ))

                print(f"💀 Pirate {self.unique_id} hijacked {merchant.unique_id}!")

                # ✅ 从调度器和商船列表里都删掉
                self.model.schedule.remove(merchant)
                merchant.pos = None
                if hasattr(self.model, "merchant_agents"):
                    self.model.merchant_agents = [
                        m for m in self.model.merchant_agents if m is not merchant
                    ]

            # 无论成功失败都进入恢复
            self.state = self.STATE_RECUP
            self.cooldown_timer = 0.0
            self.current_target_merchant = None

    def _recuperate(self, hours):
        self.cooldown_timer += hours
        if self.cooldown_timer >= self.cool_down:
            self.current_target_merchant = None
            if self.time_since_departure >= self.max_sailing_steps:
                self.state = self.STATE_RETURN
            else:
                self.state = self.STATE_SEARCH

    def _return_home(self, hours):
        self._move_towards(self.home_anchor, self.cruising_speed, hours)
        if distance(self.pos, self.home_anchor) < 1.0:
            self.time_since_departure = 0.0
            self.sailing_steps = 0
            self.state = self.STATE_SELECT

# ============================================================
# 海军（基本保持 NavyAgent.py 的写法）:contentReference[oaicite:7]{index=7}
# ============================================================
class NavyAgent(Agent):
    """
    连续空间海军（带燃料/航行步数版）：
    - 听到商船呼救马上出发
    - 任务结束一定回基地
    - 回基地途中如果还有油，可以被新的求救打断，再次前往
    - 每艘船有 max_steps 的最大航行步数，耗尽后必须先回基地加油
    """
    def __init__(self, unique_id, model,
                 speed: float = 40.0,
                 armament: float = 1.0,
                 base_pos: Tuple[float, float] | None = None,
                 max_steps: int = 200):
        super().__init__(unique_id, model)
        self.speed = speed
        self.armament = armament
        self.intercept_radius = 1.0
        self.pos_f = (0.0, 0.0)
        self.target = None
        if base_pos is None:
            base_pos = (0.0, self.model.space.y_max)
        self.base_pos = base_pos

        # 状态: idle / to_target / rtb
        self.state = "idle"

        # 燃料/步数
        self.max_steps = max_steps       # 最大能走多少步
        self.steps_left = max_steps      # 当前还能走多少步

    # ------------------ 对外接口：接收求救 ------------------
    def receive_distress(self, merchant: MerchantAgent):
        # 没油了 → 直接忽略这次呼叫
        if self.steps_left <= 0:
            return

        if merchant is None or merchant.pos is None:
            return
        if not getattr(merchant, "under_attack", False):
            return

        # 如果现在没有任务，直接接
        if self.target is None:
            self.target = merchant
            self.state = "to_target"
            return

        # 有任务 → 换成离自己更近的那个
        try:
            old_dist = distance(self.pos_f, self.target.pos)
            new_dist = distance(self.pos_f, merchant.pos)
            if new_dist < old_dist:
                self.target = merchant
                self.state = "to_target"
        except Exception:
            pass

    def can_accept_mission(self, pos: tuple[float, float]) -> bool:
        """
        判断当前海军是否可以接受新的任务。
        参数:
            pos: 任务目标的位置 (x, y)
        返回:
            True  -> 可以接受任务
            False -> 暂时无法接受
        """
        # 如果当前状态是待命 → 可以直接接
        if self.state == "idle":
            return True

        # 如果当前状态是执行任务中 → 不可接
        if self.state == "to_target":
            return False

        # 如果正在返航
        if self.state == "rtb":
            # 计算距离基地的距离
            dist_to_base = distance(self.pos_f, self.base_pos)
            # 估算返航油量阈值（剩余步数必须 > 往返消耗）
            if self.steps_left > dist_to_base / (self.speed * self.model.hours_per_step):
                return True  # 还有油，能接任务
            else:
                return False  # 油不够，必须先回去加油

        # 其他未知状态默认不接任务
        return False

    # ------------------ 主逻辑 ------------------
    def step(self):
        hours = self.model.hours_per_step
        dist_per_step = self.speed * hours

        # ① 没油了 → 不跟你讲道理，直接回去
        if self.steps_left <= 0 and self.state != "rtb":
            self.target = None
            self.state = "rtb"

        # ② 如果空闲但外面有人在挨打 → 主动出动（前提是有油）
        if self.state == "idle" and self.steps_left > 0:
            ua = [m for m in self.model.merchant_agents if getattr(m, "under_attack", False)]
            if ua:
                self.target = min(ua, key=lambda m: distance(self.pos_f, m.pos))
                self.state = "to_target"

        # ③ 去救人
        if self.state == "to_target" and self.target is not None:
            # 还没走就发现对方不挨打了 → 回去
            if not getattr(self.target, "under_attack", False):
                self.target = None
                self.state = "rtb"
                self._move_to_base(dist_per_step)
                return

            target_pos = getattr(self.target, "pos", None)
            if target_pos is None:
                # 商船可能被劫持/被删了
                self.target = None
                self.state = "rtb"
                self._move_to_base(dist_per_step)
                return

            # 真正去追
            self.pos_f = step_move(self.pos_f, target_pos, dist_per_step)
            self.model.space.move_agent(self, self.pos_f)
            self.steps_left -= 1   # 走一步扣一步

            # 到达目标附近 → 任务结束，强制回家
            if distance(self.pos_f, target_pos) <= self.intercept_radius:
                # 拦截成功就清掉标志
                if getattr(self.target, "under_attack", False):
                    self.target.under_attack = False
                if hasattr(self.model, "events"):
                    self.model.events.append(("DISRUPT", self.unique_id, self.target.unique_id))
                self.target = None
                self.state = "rtb"
            return

        # ④ 回基地
        if self.state == "rtb":
            # 返程途中如果还有油 → 可以被新的求救打断，前提是 steps_left > 0
            if self.steps_left > 0:
                ua = [m for m in self.model.merchant_agents if getattr(m, "under_attack", False)]
                if ua:
                    # 按你说的：途中又被呼叫 → 再去
                    closest = min(ua, key=lambda m: distance(self.pos_f, m.pos))
                    self.target = closest
                    self.state = "to_target"
                    return

            # 正常返航
            self._move_to_base(dist_per_step)
            self.steps_left -= 1   # 返航也要扣油

            # 到家 → 补满油，变 idle
            if distance(self.pos_f, self.base_pos) < 0.5:
                self.steps_left = self.max_steps
                self.state = "idle"
                self.target = None
            return

        # ⑤ idle 原地耗不耗油？不耗
        if self.state == "idle":
            self.model.space.move_agent(self, self.pos_f)

    # ------------------ 辅助 ------------------
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
        point_navy = (135, 120)
        point_parit = (175, 75)
        routes = [
            [port_A, port_B],
            [port_A, point_navy, port_B],
            [port_A, point_parit, port_B],
        ]
        self.ports = [port_A, port_B]

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
        # Part A和B的警戒区
        self.guard_zones = [
            {"center": port_A, "radius": 25.0, "label": "PORT_A"},
            {"center": port_B, "radius": 25.0, "label": "PORT_B"},
            {"center": (100, 150), "radius": 25.0, "label": "NAVY_BASE"},
        ]

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

    def compute_merchant_density_grid(self, grid_resolution: float = 5.0, sigma: float = 20.0):
        """
        根据商船航线生成 merchant_density_grid。

        grid_resolution: 网格间距 (nm)
        sigma: 高斯衰减距离参数 (越大越平滑)
        """
        width, height = self.space.x_max, self.space.y_max
        x_bins = np.arange(0, width + grid_resolution, grid_resolution)
        y_bins = np.arange(0, height + grid_resolution, grid_resolution)

        density_grid: Dict[Tuple[float, float], float] = {}

        for x in x_bins:
            for y in y_bins:
                point = np.array([x, y])
                density = 0.0

                # 遍历所有商船航线
                for m in self.merchant_agents:
                    route = m.route
                    for i in range(len(route) - 1):
                        p1 = np.array(route[i])
                        p2 = np.array(route[i + 1])

                        # 计算 point 到线段 p1-p2 的最短距离
                        line_vec = p2 - p1
                        p_vec = point - p1
                        line_len = np.linalg.norm(line_vec)
                        if line_len == 0:
                            dist = np.linalg.norm(p_vec)
                        else:
                            t = np.clip(np.dot(p_vec, line_vec) / (line_len**2), 0.0, 1.0)
                            proj = p1 + t * line_vec
                            dist = np.linalg.norm(point - proj)

                        # 高斯衰减
                        density += np.exp(-(dist**2) / (2 * sigma**2))

                density_grid[(x, y)] = density

        self.merchant_density_grid = density_grid

# ============================================================
# 模拟 + 画图
# ============================================================
def run_and_plot(steps=200):
    model = NavalSimModel(num_pirates=3, num_merchants=6, num_navy=1,
                          width=300, height=200,
                          hours_per_step=1/6)
    
    model.compute_merchant_density_grid(grid_resolution=5.0, sigma=20.0)

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
    # 画警戒区
    if hasattr(model, "guard_zones"):
        for gz in model.guard_zones:
            circle = plt.Circle(gz["center"], gz["radius"],
                                edgecolor="orange",
                                facecolor="none",
                                linestyle="--",
                                linewidth=1.2,
                                alpha=0.8)
            ax.add_patch(circle)
            # 标个字，方便你看哪个是哪个
            ax.text(gz["center"][0], gz["center"][1],
                    gz["label"],
                    color="orange",
                    fontsize=8,
                    ha="center",
                    va="center")

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
def run_and_animate(steps=200, interval=100):
    global ani

    model = NavalSimModel(num_pirates=3, num_merchants=6, num_navy=1,
                          width=300, height=200,
                          hours_per_step=1/6)
    for _ in range(steps):
        model.step()

    print("Hijacks:", model.hijack_count)
    print("Events:", model.events)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, model.space.x_max)
    ax.set_ylim(0, model.space.y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Maritime ABM Animation")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.grid(True, linestyle=':', alpha=0.3)

    # 先画静态航线（你现在看到的就是这个）
    for route in model.routes_template:
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, linestyle='--', color='lightgreen', linewidth=1, alpha=0.4)

    lines = {}
    scatters = {}
    max_len = 0
    for agent_id, traj in model.trajectories.items():
        max_len = max(max_len, len(traj))
        if agent_id.startswith("merchant_"):
            color, z = "green", 3
        elif agent_id.startswith("pirate_"):
            color, z = "red", 4
        else:
            color, z = "blue", 5

        line, = ax.plot([], [], color=color, linewidth=1.5, alpha=0.9, zorder=z)
        sc = ax.scatter([], [], color=color, s=30, zorder=z+1)
        lines[agent_id] = line
        scatters[agent_id] = sc

    def init():
        for line in lines.values():
            line.set_data([], [])
        for sc in scatters.values():
            # scatter 要二维的
            sc.set_offsets(np.empty((0, 2)))
        return list(lines.values()) + list(scatters.values())

    def update(frame):
        artists = []
        for agent_id, traj in model.trajectories.items():
            # 正常情况每条轨迹长度都差不多，但有的agent可能被劫走了
            last_idx = min(frame, len(traj) - 1)
            xs = [p[0] for p in traj[:last_idx+1]]
            ys = [p[1] for p in traj[:last_idx+1]]
            lines[agent_id].set_data(xs, ys)
            scatters[agent_id].set_offsets(np.array([[traj[last_idx][0], traj[last_idx][1]]]))
            artists.append(lines[agent_id])
            artists.append(scatters[agent_id])
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_len,
        init_func=init,
        interval=interval,
        blit=False,         # 关键：先关掉 blit
        repeat=False
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 想要静态图
    run_and_plot(steps=250)

    # 想要动画
    # run_and_animate(steps=250, interval=120)