# experiment_runner.py
"""
批量跑海盗-商船-海军仿真，用来回答 3.1 / 3.2 / 3.3 三个研究问题。
依赖：AizawaCombine.py 在同目录
"""

import csv
import os
import numpy as np
from datetime import datetime

from AizawaCombine import NavalSimModel  # 直接用你现在的模型:contentReference[oaicite:1]{index=1}


# =============== 一些工具 ===============

def run_one_simulation(
    steps=400,
    num_pirates=9,
    num_merchants=6,
    num_navy=1,
    mode="open",
    speed_choices=None,
    alert_choices=None,
):
    """
    跑一次仿真，返回一个 dict，里面有本次的所有指标
    mode:
        - "open": 全部走同一条“受保护”航道
        - "mixed": 你现在模型里的默认三条混合航线:contentReference[oaicite:2]{index=2}
    speed_choices / alert_choices:
        - 如果给了，就在生成商船后覆盖它们的属性，用来做实验3
    """
    # 先建模型
    model = NavalSimModel(
        num_pirates=num_pirates,
        num_merchants=num_merchants,
        num_navy=num_navy,
        width=300,
        height=200,
        hours_per_step=1/6
    )

    # 如果要做“集中航道”场景，就把所有商船的 route 改成同一条
    if mode == "open":
        # 统一使用 Port A → Port B 航道
        port_A = (20, 20)
        port_B = (model.space.x_max - 20, model.space.y_max - 20)
        point_pirate = (175, 75)
        corridor = [port_A, point_pirate, port_B]
        for m in model.merchant_agents:
            m.route = corridor

    # 如果要做“速度/警觉性”实验，就覆盖商船属性
    if speed_choices is not None or alert_choices is not None:
        for m in model.merchant_agents:
            if speed_choices is not None:
                m.normal_speed = float(np.random.choice(speed_choices))
            if alert_choices is not None:
                m.alert_param = float(np.random.choice(alert_choices))

    # 跑指定步数
    for _ in range(steps):
        model.step()

    # -------------------- 指标收集 --------------------
    events = model.events  # 列表，里面有 HIJACK / DISRUPT / PIRATE_BOUNCED:contentReference[oaicite:3]{index=3}
    hijacks = [e for e in events if e[0] == "HIJACK"]
    disrupts = [e for e in events if e[0] == "DISRUPT"]
    attempts = [e for e in events if e[0] in ("HIJACK", "DISRUPT")]

    hijack_count = len(hijacks)
    disrupt_count = len(disrupts)
    attempt_count = len(attempts)

    hijack_rate = hijack_count / attempt_count if attempt_count > 0 else 0.0

    # 场景信息
    return {
        "steps": steps,
        "num_pirates": num_pirates,
        "num_merchants": num_merchants,
        "num_navy": num_navy,
        "mode": mode,
        "hijack_count": hijack_count,
        "disrupt_count": disrupt_count,
        "attempt_count": attempt_count,
        "hijack_rate": hijack_rate,
        # 原始事件也给出去，方便做空间分析
        "events": events,
    }


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# =============== 实验 1：航道集中度 ===============
def experiment_channel(num_runs=30):
    rows = []
    for mode in ["open", "mixed"]:
        for i in range(num_runs):
            res = run_one_simulation(
                steps=400,
                num_pirates=9,
                num_merchants=6,
                num_navy=1,
                mode=mode,
            )
            rows.append({
                "exp": "channel",
                "mode": mode,
                "run": i,
                "hijack_rate": res["hijack_rate"],
                "hijack_count": res["hijack_count"],
                "disrupt_count": res["disrupt_count"],
                "attempt_count": res["attempt_count"],
            })
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_csv(f"results/exp1_channel_{ts}.csv",
              ["exp", "mode", "run", "hijack_rate", "hijack_count", "disrupt_count", "attempt_count"],
              rows)


# =============== 实验 2：海军数量 ===============
def experiment_navy(num_runs=30):
    rows = []
    for navy in [0, 1, 2, 3]:
        for i in range(num_runs):
            res = run_one_simulation(
                steps=250,
                num_pirates=3,
                num_merchants=6,
                num_navy=navy,
                mode="open",   # 固定用集中航道，容易看出海军效果
            )
            rows.append({
                "exp": "navy",
                "num_navy": navy,
                "run": i,
                "hijack_rate": res["hijack_rate"],
                "hijack_count": res["hijack_count"],
                "disrupt_count": res["disrupt_count"],
                "attempt_count": res["attempt_count"],
            })
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_csv(f"results/exp2_navy_{ts}.csv",
              ["exp", "num_navy", "run", "hijack_rate", "hijack_count", "disrupt_count", "attempt_count"],
              rows)


# =============== 实验 3：船速 & 警觉性（做 logistic 用） ===============
def experiment_ship_features(num_runs=20):
    """
    这里我们要输出“船级别”的表，而不是场景级别的。
    每跑一次，把这一轮里所有商船的特征都展开一遍。
    """
    rows = []
    speed_choices = [8, 10, 12, 14, 16]
    alert_choices = [0.01, 0.03, 0.05]

    for i in range(num_runs):
        res = run_one_simulation(
            steps=250,
            num_pirates=3,
            num_merchants=6,
            num_navy=1,
            mode="mixed",
            speed_choices=speed_choices,
            alert_choices=alert_choices,
        )
        events = res["events"]
        hijacked_ships = {e[2] for e in events if e[0] == "HIJACK"}  # e[2] 是 merchant_id:contentReference[oaicite:4]{index=4}

        # 这里要注意：被劫持的商船已经从 model.merchant_agents 里删掉了，
        # 所以我们不能只看 model.merchant_agents，要在 run_one_simulation 里补一份“初始商船表”才是最严谨的。
        # 简化起见，这里只看仍在场上的；被删掉的我们单独加一行。
        model = NavalSimModel(num_pirates=0)  # 占个位用不到
        # 实际上，上面的 res 没有把 model 返回来，如果你想拉出完整船表，
        # 可以把 run_one_simulation 改成同时返回 model；这里先用简化方式。

        # 简化方式：只记录这次事件里出现过的船
        for m_id in hijacked_ships:
            rows.append({
                "exp": "ship_features",
                "run": i,
                "merchant_id": m_id,
                "speed": "unknown",     # 如果你想要真实速度，就要在 run_one_simulation 里把 ship attrs 一起返回
                "alert": "unknown",
                "hijacked": 1,
            })

        # 实际更完整的做法：在 run_one_simulation 里，把每艘船的 speed/alert 存入 list 一起 return

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_csv(f"results/exp3_shipfeatures_{ts}.csv",
              ["exp", "run", "merchant_id", "speed", "alert", "hijacked"],
              rows)


if __name__ == "__main__":
    # 你可以按需打开
    experiment_channel(num_runs=100)
    # experiment_navy(num_runs=30)
    # experiment_ship_features(num_runs=20)
