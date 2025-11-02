from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random

# --- 1. Agent (商船代理) ---
class MerchantAgent(Agent):
    """
    一个沿着固定航线航行的商船代理。
    它有两档速度，并根据视野内的威胁计算警戒等级。
    """
    def __init__(self, unique_id, model, route, speed_normal=1, speed_escape=2, max_vision=5):
        super().__init__(unique_id, model)

        # --- 固定参数 ---
        self.speed_normal = speed_normal
        self.speed_escape = speed_escape
        self.max_vision = max_vision
        self.route = route  # 预设的航线列表 [(x1, y1), (x2, y2), ...]

        # --- 状态参数 ---
        self.route_index = 0             # 当前目标航点在 route 列表中的索引
        self.current_speed = speed_normal # 初始速度为正常速度
        self.alert_level = 0.0           # 初始警戒等级
        self.at_port = True              # 初始停靠在起点

        # 初始位置设置为航线起点
        self.pos = self.route[0]

    def calculate_alertness(self):
        """
        根据视野范围内距离最近的威胁，计算警戒等级。
        警戒参数根据距离线性增长（距离越近，警戒等级越高）。

        注意：此处假设模型中存在名为 'ThreatAgent' 的威胁代理。
        """
        min_distance = float('inf')
        threat_found = False

        # 获取视野范围内的所有单元格
        # Mesa 2.x 提供了 get_neighborhood 配合 get_cell_list_contents

        # 1. 查找所有代理
        all_agents = self.model.schedule.agents

        # 2. 过滤出威胁代理 (假设ThreatAgent已定义)
        threat_agents = [a for a in all_agents if isinstance(a, ThreatAgent)]

        # 3. 计算与最近威胁的距离
        if threat_agents:
            for threat in threat_agents:
                # 使用 Chebyshev 距离 (最大坐标差) 作为网格距离
                distance = self.model.grid.get_distance(self.pos, threat.pos)
                if distance <= self.max_vision:
                    min_distance = min(min_distance, distance)
                    threat_found = True

        # 4. 计算警戒等级
        if threat_found:
            # 线性增长：距离越近，警戒等级越高 (0.0 到 1.0)
            # 距离为 0 时警戒为 1.0，距离为 max_vision 时警戒为 0.0
            self.alert_level = 1.0 - (min_distance / self.max_vision)

            # 如果警戒等级超过某个阈值，切换到逃命速度
            if self.alert_level >= 0.5:
                self.current_speed = self.speed_escape
            else:
                self.current_speed = self.speed_normal
        else:
            self.alert_level = 0.0
            self.current_speed = self.speed_normal


    def follow_route(self):
        """
        商船以当前速度向航线的下一个目标点移动。
        """
        target_pos = self.route[self.route_index]

        # 如果已到达目标航点，更新目标索引
        if self.pos == target_pos:
            self.at_port = True
            # 切换到下一个目标点，循环航线
            self.route_index = (self.route_index + 1) % len(self.route)
            target_pos = self.route[self.route_index]

        else:
            self.at_port = False

        # --- 计算移动方向和步数 ---
        current_x, current_y = self.pos
        target_x, target_y = target_pos
        speed = self.current_speed

        # 计算 x 和 y 方向的距离
        dx = target_x - current_x
        dy = target_y - current_y

        # 计算移动步长 (最多移动 speed 步)
        # 优先沿着最长的距离轴移动，确保是朝着目标方向

        # X 轴移动：不超过 speed，且方向正确
        step_x = 0
        if dx != 0:
            step_x = min(abs(dx), speed) * (1 if dx > 0 else -1)

        # Y 轴移动：在剩余速度内，且方向正确
        remaining_speed = speed - abs(step_x) # 假设只在一条轴上移动
        step_y = 0
        if remaining_speed > 0 and dy != 0:
             step_y = min(abs(dy), remaining_speed) * (1 if dy > 0 else -1)

        # 如果 remaining_speed 仍为 0，且目标未到，则尝试在Y轴移动
        if step_x == 0 and dy != 0:
            step_y = min(abs(dy), speed) * (1 if dy > 0 else -1)

        new_x = current_x + step_x
        new_y = current_y + step_y

        # Mesa 移动
        new_pos = (new_x, new_y)
        self.model.grid.move_agent(self, new_pos)

    def step(self):
        self.calculate_alertness() # 先侦测威胁并调整速度
        self.follow_route()        # 再根据航线和速度移动