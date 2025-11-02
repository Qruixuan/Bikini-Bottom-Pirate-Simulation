# naval_model.py

import math, random
from typing import Tuple, List
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

# --- 导入 Agent 类 ---
# 假设 MerchantShip.py 中包含了 MerchantAgent 和 distance 函数
from MerchantShip import MerchantAgent, distance
# 假设 pirate.py 中包含了 PirateAgent 类
from pirate import PirateAgent


# ====================================================================
# --- ⚓ NavalSimModel (模拟模型) ---
# ====================================================================

class NavalSimModel(Model):
    """海盗和商船模拟模型."""

    def __init__(self, width=300, height=200, num_pirates=3, num_merchants=5, hours_per_step=1):
        super().__init__()
        self.space = ContinuousSpace(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.hours_per_step = hours_per_step

        # 将边界信息暴露给 Agent
        self.space.x_max = width
        self.space.y_max = height

        # 统计指标
        self.hijack_count = 0

        # 全局地图信息
        # 海盗根据这个网格来选择目标区域
        self.merchant_density_grid = {(200, 150): 5, (50, 50): 3}
        self.navy_positions = []  # 预留给军舰位置

        # --- 1. 定义航线 (Port 1 <-> Port 2 <-> Port 3) ---
        port_A = (20, 20)
        port_B = (width - 20, height - 20)
        port_C = (width // 2, height // 2)

        # 航线定义 (循环)
        route_1 = [port_A, port_B, port_A]
        route_2 = [port_B, port_C, port_A, port_B]
        self.all_routes = [route_1, route_2]

        # --- 2. 添加商船 ---
        for i in range(num_merchants):
            route = random.choice(self.all_routes)
            # 确保商船从航线起点开始
            start_pos = route[0] if route else (width / 2, height / 2)

            merchant = MerchantAgent(
                f"merchant_{i}",
                self,
                route=route,
                normal_speed_kn=random.uniform(10, 14),
                evasion_speed_kn=random.uniform(16, 20),
                visibility_nm=15,
                alert_param=0.03
            )
            self.space.place_agent(merchant, start_pos)
            self.schedule.add(merchant)

        # --- 3. 添加海盗 ---
        for i in range(num_pirates):
            # 海盗基地设在地图一角
            home = (random.uniform(10, width * 0.1), random.uniform(0, height * 0.2))
            pirate = PirateAgent(f"pirate_{i}", self, home_anchor=home)
            self.space.place_agent(pirate, home)
            self.schedule.add(pirate)

    def step(self):
        # 每次步进都随机执行所有 Agent 的 step
        self.schedule.step()

        # 模拟结束条件：所有商船都被抢劫
        merchant_count = sum(1 for a in self.schedule.agents if isinstance(a, MerchantAgent))
        if merchant_count == 0 and len(self.schedule.agents) > 0:
            self.running = False


# ====================================================================
# --- 运行模拟 ---
# ====================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- 模拟参数 ---
    SIM_STEPS = 500
    GRID_WIDTH = 300
    GRID_HEIGHT = 200

    model = NavalSimModel(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_pirates=3,
        num_merchants=10,
        hours_per_step=1
    )

    # --- 数据收集 ---
    data_collector = {
        "step": [],
        "pirate_positions": {p.unique_id: [] for p in model.schedule.agents if isinstance(p, PirateAgent)},
        "merchant_positions": {m.unique_id: [] for m in model.schedule.agents if isinstance(m, MerchantAgent)},
        "hijacks": []
    }

    # --- 运行循环 ---
    for step in range(SIM_STEPS):
        if not model.running:
            break

        model.step()
        data_collector["step"].append(step)
        data_collector["hijacks"].append(model.hijack_count)

        # 记录位置
        for agent in model.schedule.agents:
            if isinstance(agent, PirateAgent):
                data_collector["pirate_positions"][agent.unique_id].append(agent.pos)
            elif isinstance(agent, MerchantAgent):
                data_collector["merchant_positions"][agent.unique_id].append(agent.pos)

    print("\n✅ 模拟结束。")
    print(f"总运行步数: {step}")
    print(f"总劫持次数: {model.hijack_count}")

    # --- 结果可视化 ---

    plt.figure(figsize=(10, 6))

    # 绘制海盗轨迹
    for pid, positions in data_collector["pirate_positions"].items():
        if positions:
            xs, ys = zip(*positions)
            plt.plot(xs, ys, '-', alpha=0.6, linewidth=2, label=f"Pirate {pid}")

    # 绘制商船轨迹 (只绘制存活的，或直到被移除)
    for mid, positions in data_collector["merchant_positions"].items():
        if positions:
            xs, ys = zip(*positions)
            plt.plot(xs, ys, '--', alpha=0.4, linewidth=1, label=f"Merchant {mid}" if mid == "merchant_0" else None)

    # 绘制起始点/港口
    plt.scatter(20, 20, c='green', marker='s', s=100, label="Port A")
    plt.scatter(GRID_WIDTH - 20, GRID_HEIGHT - 20, c='green', marker='s', s=100, label="Port B")

    plt.title("海盗与商船模拟轨迹 (Mesa 2.x)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(0, GRID_WIDTH)
    plt.ylim(0, GRID_HEIGHT)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()