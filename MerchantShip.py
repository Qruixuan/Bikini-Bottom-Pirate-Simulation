import math, random
from typing import Tuple, List
from mesa import Agent


# --- Utility function ---
def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """计算两点之间的欧几里得距离."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


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