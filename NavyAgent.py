"""
Single-file maritime ABM:
- MerchantAgent
- PirateAgent
- NavyAgent
- NavalSimModel (Mesa Model)

依赖:
    pip install mesa
"""

import math
import random
from typing import Tuple, List

from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation


# ============================================================
# 小工具
# ============================================================
def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
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
# 商船
# ============================================================
class MerchantAgent(Agent):
    STATE_SAILING = "sailing"
    STATE_EVADING = "evading"
    STATE_IN_PORT = "in_port"

    def __init__(
            self,
            unique_id,
            model,
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
        self.visibility = visibility_nm  # 海盗被发现的半径
        self.alert_param = alert_param
        self.port_wait_time = port_wait_time_hrs

        self.state = self.STATE_SAILING
        self.current_route_index = 0
        self.wait_timer = 0.0
        self.awareness = False
        self.last_known_pirate_pos = None

        # 给海军看的状态
        self.under_attack = False

    # ----------------- 内部工具 -----------------
    def _get_current_target(self) -> Tuple[float, float]:
        if not self.route or self.current_route_index >= len(self.route):
            return self.pos
        return self.route[self.current_route_index]

    def _move_towards(self, dest, speed_kn, hours):
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

        x_max = getattr(self.model.space, 'x_max', self.model.space.width)
        y_max = getattr(self.model.space, 'y_max', self.model.space.height)

        clamped_x = max(0.0, min(new_pos[0], x_max))
        clamped_y = max(0.0, min(new_pos[1], y_max))
        final_pos = (clamped_x, clamped_y)

        if final_pos != new_pos:
            arrived = False

        self.model.space.move_agent(self, final_pos)
        return arrived

    # ----------------- 呼救 -----------------
    def receive_distress(self, pirate_pos):
        """
        被攻击/发现海盗时调用：
        1. 标记自己 under_attack
        2. 通知所有海军
        """
        self.under_attack = True
        if hasattr(self.model, "navy_agents"):
            for n in self.model.navy_agents:
                n.receive_distress(self)

    # ----------------- 每一步 -----------------
    def step(self):
        hours = self.model.hours_per_step

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
                # 已经安全
                self.state = self.STATE_SAILING
                self.last_known_pirate_pos = None
                self.under_attack = False
                self._sail_route(hours)

        elif self.state == self.STATE_IN_PORT:
            self._wait_in_port(hours)

    # ----------------- 细分逻辑 -----------------
    def _check_pirate_threat(self):
        closest_pirate_pos = None
        min_distance = float('inf')

        for agent in self.model.schedule.agents:
            if isinstance(agent, PirateAgent):
                d = euclid(self.pos, agent.pos)
                if d <= self.visibility:
                    # 简单感知概率
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

        escape_dest = (self.pos[0] + dx / d * escape_distance,
                       self.pos[1] + dy / d * escape_distance)

        self._move_towards(escape_dest, self.evasion_speed, hours)

        # 逃跑时也要继续呼救
        self.receive_distress(pirate_pos)

    def _wait_in_port(self, hours):
        self.wait_timer += hours
        if self.wait_timer >= self.port_wait_time:
            self.current_route_index = 0
            self.state = self.STATE_SAILING
            self.under_attack = False


# ============================================================
# 海盗
# ============================================================
class PirateAgent(Agent):
    """
    简化版海盗：
    - 在海区里乱晃 / 搜索
    - 靠近商船就“攻击”
    - 攻击时会让商船呼救
    """
    def __init__(self, unique_id, model, home_anchor=(10, 10), attack_radius=2.0):
        super().__init__(unique_id, model)
        self.home_anchor = home_anchor
        self.attack_radius = attack_radius

    def step(self):
        # 简单游走
        jitter = (random.uniform(-2, 2), random.uniform(-2, 2))
        new_pos = (self.pos[0] + jitter[0], self.pos[1] + jitter[1])

        x_max = getattr(self.model.space, 'x_max', self.model.space.width)
        y_max = getattr(self.model.space, 'y_max', self.model.space.height)
        new_pos = (max(0, min(new_pos[0], x_max)),
                   max(0, min(new_pos[1], y_max)))

        self.model.space.move_agent(self, new_pos)

        # 找最近的商船
        nearest_m = None
        nearest_d = 999999
        for m in self.model.merchant_agents:
            d = euclid(self.pos, m.pos)
            if d < nearest_d:
                nearest_d = d
                nearest_m = m

        if nearest_m and nearest_d <= self.attack_radius:
            # 攻击到了
            nearest_m.receive_distress(self.pos)
            # 这里可以标 hijack，如果海军没来得及
            # 也可以放在 model 里检测


# ============================================================
# 海军
# ============================================================
class NavyAgent(Agent):
    """
    连续空间海军：
    - 速度固定 40 km/h
    - 武装 = 1.0
    - 听到商船呼救马上出发
    - 商船会动 → 每一步都追最新位置
    - 拦截后回左上角基地
    - 回去途中也能被新的 distress 打断
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

        try:
            if euclid(self.pos_f, merchant.pos) < euclid(self.pos_f, self.target.pos):
                self.target = merchant
                self.state = "to_target"
        except Exception:
            pass

    def step(self):
        hours = self.model.hours_per_step
        dist_per_step = self.speed * hours

        # 主动找一找有没有被打的
        if self.state != "to_target" and self.target is None:
            ua = [m for m in self.model.merchant_agents if getattr(m, "under_attack", False)]
            if ua:
                self.target = min(ua, key=lambda m: euclid(self.pos_f, m.pos))
                self.state = "to_target"

        # 去救人
        if self.state == "to_target" and self.target is not None:
            if not getattr(self.target, "under_attack", False):
                # 目标已经没事了 → 回家
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

            if euclid(self.pos_f, target_pos) <= self.intercept_radius:
                # 成功拦截
                self.target.under_attack = False
                if hasattr(self.model, "events"):
                    self.model.events.append(("DISRUPT", self.unique_id, self.target.unique_id))
                self.target = None
                self.state = "rtb"
            return

        # 回基地
        if self.state == "rtb":
            self._move_to_base(dist_per_step)
            if euclid(self.pos_f, self.base_pos) < 0.5:
                self.state = "idle"
            return

        # idle
        if self.state == "idle":
            self.model.space.move_agent(self, self.pos_f)

    def _move_to_base(self, dist):
        self.pos_f = step_move(self.pos_f, self.base_pos, dist)
        self.model.space.move_agent(self, self.pos_f)


# ============================================================
# 模型，把三种 agent 融合在一起
# ============================================================
class NavalSimModel(Model):
    def __init__(self,
                 width=300, height=200,
                 num_pirates=3,
                 num_merchants=8,
                 num_navy=1,
                 hours_per_step=1/6):
        super().__init__()
        self.space = ContinuousSpace(width, height, torus=False)
        self.space.x_max = width
        self.space.y_max = height

        self.schedule = RandomActivation(self)
        self.hours_per_step = hours_per_step

        # 数据收集字段
        self.events: list[tuple] = []
        self.hijack_count = 0

        # 这两个是给别的 agent 快速访问的
        self.merchant_agents: list[MerchantAgent] = []
        self.navy_agents: list[NavyAgent] = []
        self.navy_positions: list[Tuple[float, float]] = []

        # -------- 航线随便造两条 --------
        port_A = (20, 20)
        port_B = (width - 20, height - 20)
        port_C = (width // 2, height // 2)
        routes = [
            [port_A, port_B, port_A],
            [port_B, port_C, port_A, port_B],
        ]

        # 1) 商船
        for i in range(num_merchants):
            route = random.choice(routes)
            start_pos = (20, 20)
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

        # 2) 海盗
        for i in range(num_pirates):
            home = (random.uniform(10, width * 0.3), random.uniform(0, height * 0.4))
            p = PirateAgent(f"pirate_{i}", self, home_anchor=home, attack_radius=2.0)
            self.space.place_agent(p, home)
            self.schedule.add(p)

        # 3) 海军
        for i in range(num_navy):
            eps = 0.001
            base = (0.0, height - eps)  # 不要写 (0, height)
            n = NavyAgent(f"navy_{i}", self, base_pos=base)
            n.pos_f = base
            self.space.place_agent(n, base)
            self.schedule.add(n)
            self.navy_agents.append(n)

    def step(self):
        # 更新海军位置给需要的 agent 看
        self.navy_positions = [n.pos for n in self.navy_agents]
        self.schedule.step()


# ============================================================
# demo 跑一下
# ============================================================
if __name__ == "__main__":
    model = NavalSimModel(num_pirates=3, num_merchants=6, num_navy=2, hours_per_step=1/6)

    for t in range(200):
        model.step()

    print("Events:", model.events)
