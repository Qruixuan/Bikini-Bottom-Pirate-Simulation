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
        # 【最终修正点 1：检查 Agent 是否仍在空间中】
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
        """响应求救信号，供海盗调用."""
        pass

    def step(self):
        hours = self.model.hours_per_step

        # 【修正点 2：调度安全检查】如果 Agent 不在调度器中，立即停止运行
        if self not in self.model.schedule.agents:
            return

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
                self.state = self.STATE_SAILING
                self.last_known_pirate_pos = None
                self._sail_route(hours)

        elif self.state == self.STATE_IN_PORT:
            self._wait_in_port(hours)

    # --- 内部行为方法 ---

    def _check_pirate_threat(self):
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
            return

        pirate_pos = self.last_known_pirate_pos
        dx, dy = self.pos[0] - pirate_pos[0], self.pos[1] - pirate_pos[1]

        x_max = getattr(self.model.space, 'x_max', 1000)
        y_max = getattr(self.model.space, 'y_max', 1000)
        escape_distance = x_max + y_max

        d = math.hypot(dx, dy)
        if d == 0: return

        escape_dest = (self.pos[0] + dx / d * escape_distance, self.pos[1] + dy / d * escape_distance)

        self._move_towards(escape_dest, self.evasion_speed, hours)

        self.receive_distress(pirate_pos)

    def _wait_in_port(self, hours):
        self.wait_timer += hours
        if self.wait_timer >= self.port_wait_time:
            self.current_route_index = 0
            self.state = self.STATE_SAILING