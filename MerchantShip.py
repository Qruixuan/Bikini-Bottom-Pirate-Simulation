import math, random
from typing import Tuple, List
from mesa import Agent


# --- Utility function (MUST BE INCLUDED) ---
def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """计算两点之间的欧几里得距离."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class MerchantAgent(Agent):
    """一个有固定航线、动态警戒和两档速度的商船."""
    STATE_SAILING = "sailing"
    STATE_EVADING = "evading"  # 新增状态：逃避
    STATE_IN_PORT = "in_port"  # 保持或根据需求移除

    def __init__(
            self, unique_id, model,
            route: List[Tuple[float, float]],  # 固定航线：一系列坐标点
            normal_speed_kn=12,
            evasion_speed_kn=18,  # 新增：逃命速度
            visibility_nm=8,  # 新增：商船视野范围
            alert_param=0.05,  # 新增：警戒增长参数 (per unit distance)
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
            # 航线结束，返回当前位置保持静止或循环航线
            return self.pos
        return self.route[self.current_route_index]

    def _move_towards(self, dest, speed_kn, hours):
        """移动逻辑（与海盗部分相似，可复用）."""
        step = speed_kn * hours
        cur = self.pos
        dx, dy = dest[0] - cur[0], dest[1] - cur[1]
        d = math.hypot(dx, dy)
        if d <= step or d == 0:
            new_pos = dest
            arrived = True
        else:
            # 避免除以零，虽然 d==0 的情况在上面已处理
            new_pos = (cur[0] + dx / d * step, cur[1] + dy / d * step)
            arrived = False
        self.model.space.move_agent(self, new_pos)
        return arrived

    def step(self):
        hours = self.model.hours_per_step

        # 1. 检查附近海盗并更新警戒
        pirate_threat = self._check_pirate_threat()

        if self.state == self.STATE_SAILING:
            if pirate_threat:
                # 发现威胁，切换到逃避状态
                self.state = self.STATE_EVADING
                self.last_known_pirate_pos = pirate_threat
            else:
                self._sail_route(hours)

        elif self.state == self.STATE_EVADING:
            if pirate_threat:
                # 仍在威胁范围内，继续逃跑
                self.last_known_pirate_pos = pirate_threat
                self._evade(hours)
            else:
                # 逃脱成功，恢复航行
                self.state = self.STATE_SAILING
                self.last_known_pirate_pos = None
                # 恢复航行一小步，但不再是逃命速度
                self._sail_route(hours)

        elif self.state == self.STATE_IN_PORT:
            self._wait_in_port(hours)

    # --- 内部行为方法 (已修正缩进) ---

    def _check_pirate_threat(self):
        closest_pirate_pos = None
        min_distance = float('inf')

        for agent in self.model.schedule.agents:
            # 仅检查海盗 (假设 PirateAgent, 检查类名更安全)
            if agent.__class__.__name__ == 'PirateAgent':
                d = distance(self.pos, agent.pos)

                # 在视野范围内
                if d <= self.visibility:
                    # 警戒计算：警戒阈值 = 1.0 - distance * alert_param
                    # 距离越近，阈值越高
                    alert_threshold = 1.0 - d * self.alert_param
                    alert_threshold = max(0.0, min(1.0, alert_threshold))  # 确保在 [0, 1] 范围内

                    # 随机判断是否触发警戒
                    if random.random() < alert_threshold:
                        # 触发警戒，将视野内的海盗记录为威胁
                        if d < min_distance:
                            min_distance = d
                            closest_pirate_pos = agent.pos
                            self.awareness = True  # 一旦触发警戒，设置 awareness

        return closest_pirate_pos

    def _sail_route(self, hours):
        target = self._get_current_target()
        arrived = self._move_towards(target, self.normal_speed, hours)

        if arrived:
            # 移动到航线下一个点
            self.current_route_index += 1
            if self.current_route_index >= len(self.route):
                # 航线结束，进入 IN_PORT 状态
                if self.route:
                    self.state = self.STATE_IN_PORT
                else:
                    self.state = self.STATE_SAILING  # 保持在原地

    def _evade(self, hours):
        if self.last_known_pirate_pos is None:
            self.state = self.STATE_SAILING
            return

        pirate_pos = self.last_known_pirate_pos

        # 计算逃跑方向（反方向）
        dx, dy = self.pos[0] - pirate_pos[0], self.pos[1] - pirate_pos[1]

        # 设定一个足够远的逃跑目标点
        # 逃到地图外，使用 model.space 的边界
        x_max = getattr(self.model.space, 'x_max', 1000)
        y_max = getattr(self.model.space, 'y_max', 1000)
        escape_distance = x_max + y_max

        # 归一化方向向量并乘以逃跑距离
        d = math.hypot(dx, dy)
        if d == 0: return  # 避免除零

        escape_dest = (self.pos[0] + dx / d * escape_distance, self.pos[1] + dy / d * escape_distance)

        # 以逃命速度移动
        self._move_towards(escape_dest, self.evasion_speed, hours)

        # 您的海盗代码中有一个 receive_distress 方法，这里调用它来模拟发送求救信号
        # 注意：MerchantAgent 初始代码中没有这个方法，如果需要，请确保在 MerchantAgent 中定义它。
        # 假设原 MerchantAgent 有这个方法：
        # self.receive_distress(pirate_pos)

    def _wait_in_port(self, hours):
        self.wait_timer += hours
        if self.wait_timer >= self.port_wait_time:
            # 重新开始航线或循环航线
            self.current_route_index = 0
            self.state = self.STATE_SAILING
