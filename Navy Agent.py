import math, random
from typing import Tuple
from mesa import Model, Agent
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def step_move(pos: Tuple[float, float], target: Tuple[float, float], dist: float) -> Tuple[float, float]:
    """从 pos 朝 target 走 dist 这么远，返回新坐标（连续空间版本）"""
    x, y = pos
    tx, ty = target
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy) or 1e-9
    if dist >= d:
        return (tx, ty)
    return (x + dist * dx / d, y + dist * dy / d)

class Navy(Agent):
    """
    连续空间版 Navy
    - 速度固定 40 km/h
    - 武装 = 1.0
    - 听到商船呼救就锁定最近的在遭攻击的商船
    - 每一步都朝“商船当前的位置”移动（商船动它也跟着动）
    - 距离 <= 1 km 时判定为成功拦截，写入 model.events
    """
    def __init__(self, model, speed: float = 40.0, armament: float = 1.0, name: str = None):
        super().__init__(model)
        self.name = name
        self.speed = speed
        self.armament = armament
        self.target = None     # 目标商船
        self.pos_f = (0.0, 0.0)  # 连续坐标，用来自己算移动
        self.intercept_radius = 1.0

    def receive_distress(self, merchant):
        """被商船叫到，登记成目标；如果已有目标，则换成更近的那个。"""
        if merchant is None:
            return
        if not getattr(merchant, "under_attack", False):
            return

        if self.target is None:
            self.target = merchant
        else:
            # 换更近的
            try:
                if euclid(self.pos_f, merchant.pos) < euclid(self.pos_f, self.target.pos):
                    self.target = merchant
            except Exception:
                pass

    def step(self):
        # 如果还没目标，就主动扫一遍模型里被攻击的商船
        if self.target is None:
            under_attacks = [
                m for m in getattr(self.model, "merchant_agents", [])
                if getattr(m, "under_attack", False)
            ]
            if under_attacks:
                self.target = min(under_attacks, key=lambda m: euclid(self.pos_f, m.pos))

        # 有目标并且目标还在被打
        if self.target is not None and getattr(self.target, "under_attack", False):
            # 目标当前位置（连续空间用 self.target.pos）
            target_pos = getattr(self.target, "pos", None)
            if target_pos is None:
                # 找不到就放弃
                self.target = None
            else:
                # 计算本步能走的距离
                dist_per_step = self.speed * self.model.step_time  # km per step
                # 自己的连续坐标前进
                self.pos_f = step_move(self.pos_f, target_pos, dist_per_step)
                # 同步到 ContinuousSpace
                self.model.space.move_agent(self, self.pos_f)

                # 拦截判定
                if euclid(self.pos_f, target_pos) <= self.intercept_radius:
                    # armament=1 -> 总是成功
                    navy_id = self.name if self.name is not None else self.unique_id
                    merchant_id = getattr(self.target, "name", self.target.unique_id)
                    # 记录事件
                    if hasattr(self.model, "events"):
                        self.model.events.append(("DISRUPT", navy_id, merchant_id))
                    # 取消商船的 under_attack
                    try:
                        self.target.under_attack = False
                    except Exception:
                        pass
                    # 自己也把目标清掉
                    self.target = None
        else:
            # 没目标就保持原地，也要同步一次位置
            self.model.space.move_agent(self, self.pos_f)

    def advance(self):
        pass
