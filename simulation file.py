import random
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
    evasion_choices=None,     # ← 只控制“逃逸速度”
    alert_choices=None,
    baseline_evasion=None,  # ← 新增：给所有船固定一个基线逃逸速度
    baseline_alert=None,  # ← 新增：给所有船固定一个基线警戒度
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
    if mode == "mixed":
        # 分散航道：给每艘船随机一条不同的/更绕的路线
        port_A = (20, 20)
        port_B = (model.space.x_max - 20, model.space.y_max - 20)
        point_navy = (135, 120)
        point_pirate = (175, 75)

        candidate_routes = [
            [port_A, port_B],
            [port_A, point_navy, port_B],
            [port_A, point_pirate, port_B],
        ]
        for m in model.merchant_agents:
            # ⚠️ 这里要用 random.choice，不能用 np.random.choice
            route = random.choice(candidate_routes)
            # 拷一份，别所有船共用同一个 list
            m.route = list(route)

    # 如果要做“速度/警觉性”实验，就覆盖商船属性
    ship_records = {}
    for m in model.merchant_agents:
        # 基线逃逸速度
        if baseline_evasion is not None:
            if hasattr(m, "evasion_speed"):
                m.evasion_speed = float(baseline_evasion)
            elif hasattr(m, "evasion_speed_kn"):
                m.evasion_speed_kn = float(baseline_evasion)
        # 基线警戒度
        if baseline_alert is not None:
            m.alert_param = float(baseline_alert)

        # 随机覆盖（若给了 choices）
        if evasion_choices is not None:
            v = float(np.random.choice(evasion_choices))
            if hasattr(m, "evasion_speed"):
                m.evasion_speed = v
            elif hasattr(m, "evasion_speed_kn"):
                m.evasion_speed_kn = v
        if alert_choices is not None:
            a = float(np.random.choice(alert_choices))
            m.alert_param = a

        ship_records[m.unique_id] = {
            "merchant_id": m.unique_id,
            "evasion_speed": float(getattr(m, "evasion_speed",
                                           getattr(m, "evasion_speed_kn", 0.0))),
            "alert": float(getattr(m, "alert_param", 0.0)),
        }

    # —— 跑模拟 —— #
    for _ in range(steps):
        model.step()

    # —— 指标&逐船 —— #
    events = model.events
    hijacks = [e for e in events if e[0] == "HIJACK"]
    disrupts = [e for e in events if e[0] == "DISRUPT"]
    attempt_count = getattr(model, "attempt_count", 0)

    hijack_count = len(hijacks)
    disrupt_count = len(disrupts)
    hijack_rate = hijack_count / attempt_count if attempt_count > 0 else 0.0

    hijacked_ids = {e[2] for e in hijacks}
    per_ship = [{
        "merchant_id": mid,
        "evasion_speed": rec["evasion_speed"],
        "alert": rec["alert"],
        "hijacked": 1 if mid in hijacked_ids else 0,
    } for mid, rec in ship_records.items()]

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
        "events": events,
        "per_ship": per_ship,
    }

def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 写入数据
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# =============== 实验 1：航道集中度 ===============
def experiment_channel(num_runs=100):
    rows = []
    for mode in ["open", "mixed"]:
        for i in range(num_runs):
            res = run_one_simulation(
                steps=400,
                num_pirates=6,
                num_merchants=9,
                num_navy=9,
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
def experiment_navy(num_runs=100):
    rows = []
    for navy in [0, 1, 2, 3]:
        for i in range(num_runs):
            res = run_one_simulation(
                steps=400,
                num_pirates=9,
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
def experiment_evasion_only(num_runs=20):
    rows = []
    evasion_choices = [14, 16, 18, 20, 22]  # 你要考察的速度档
    baseline_alert = 0.0                   # 固定警戒度

    for i in range(num_runs):
        res = run_one_simulation(
            steps=400,
            num_pirates=9,
            num_merchants=6,
            num_navy=1,
            mode="open",
            evasion_choices=evasion_choices,   # ← 只随机速度
            alert_choices=None,                # ← 不随机警戒度
            baseline_alert=baseline_alert,     # ← 全部同一基线警戒
        )
        for row in res["per_ship"]:
            rows.append({
                "exp": "evasion_only",
                "run": i,
                "merchant_id": row["merchant_id"],
                "evasion_speed": row["evasion_speed"],
                "alert": baseline_alert,
                "hijacked": row["hijacked"],
            })

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_csv(
        f"results/exp3_evasion_only_{ts}.csv",
        ["exp", "run", "merchant_id", "evasion_speed", "alert", "hijacked"],
        rows
    )

def experiment_alert_only(num_runs=20):
    rows = []
    alert_choices    = [0.01, 0.03, 0.05]  # 你要考察的警戒档
    baseline_evasion = 18                  # 固定逃逸速度（单位与模型一致）

    for i in range(num_runs):
        res = run_one_simulation(
            steps=400,
            num_pirates=9,
            num_merchants=6,
            num_navy=1,
            mode="open",
            evasion_choices=None,              # ← 不随机速度
            alert_choices=alert_choices,       # ← 只随机警戒度
            baseline_evasion=baseline_evasion, # ← 全部同一基线速度
        )
        for row in res["per_ship"]:
            rows.append({
                "exp": "alert_only",
                "run": i,
                "merchant_id": row["merchant_id"],
                "evasion_speed": baseline_evasion,
                "alert": row["alert"],
                "hijacked": row["hijacked"],
            })

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_csv(
        f"results/exp4_alert_only_{ts}.csv",
        ["exp", "run", "merchant_id", "evasion_speed", "alert", "hijacked"],
        rows
    )

if __name__ == "__main__":
    # 你可以按需打开
    experiment_channel(num_runs=200)
    # experiment_navy(num_runs=100)
    # experiment_evasion_only(num_runs=200)
    # experiment_alert_only(num_runs=1000)
