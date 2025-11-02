import math
from typing import Tuple
from mesa import Agent


# ========== 工具函数 ==========
def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """两点欧氏距离"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def step_move(pos: Tuple[float, float],
              target: Tuple[float, float],
              dist: float) -> Tuple[float, float]:
    """
    从 pos 朝 target 走 dist 这么远（连续空间版本）
    如果 dist 超过了剩余距离，就直接走到 target
    """
    x, y = pos
    tx, ty = target
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy) or 1e-9
    if dist >= d:
        return (tx, ty)
    return (x + dist * dx / d, y + dist * dy / d)


# ========== 海军智能体 ==========
class Navy(Agent):
    """
    连续空间版 Navy

    需求点全覆盖：
    1. 速度永远是 40 km/h
    2. 武装程度是 1（100% 成功）
    3. 听到商船呼救就马上出发
    4. 商船在移动 → 海军每一步都重新追踪商船当前位置
    5. 拦截完成以后回到基地
    6. 基地在地图左上角
    7. 回基地途中，如果又有新的求救，可以被打断再去救

    注意：
    - 要求 model 里有：space（ContinuousSpace）、merchant_agents（list）、step_time（小时制，比如 1/6）
    - 建议 model 里有：events = [] 方便统计 ("DISRUPT", navy_id, merchant_id)
    """

    def __init__(self, model,
                 speed: float = 40.0,
                 armament: float = 1.0,
                 name: str | None = None):
        super().__init__(model)
        self.name = name
        self.speed = speed            # 固定 40
        self.armament = armament      # 固定 1
        self.intercept_radius = 1.0   # 1 km 内算拦截成功

        # 连续坐标（不要在这里读 self.pos，因为 agent 还没被放进 space）
        self.pos_f: Tuple[float, float] = (0.0, 0.0)

        # 当前锁定的商船
        self.target = None  # -> Merchant

        # 基地：地图左上角
        # ContinuousSpace(0,0) 默认在左下，所以左上是 (0, space.height)
        # 如果你模型坐标是别的方向，就把它改成 (0.0, 0.0) 就行
        self.base_pos: Tuple[float, float] = (0.0, self.model.space.height)

        # 状态机：idle / to_target / rtb
        self.state: str = "idle"

    # =========================================================
    # 外部调用：商船在被打时应该调用 navy.receive_distress(self)
    # =========================================================
    def receive_distress(self, merchant):
        """
        商船呼救进来：
        - 如果我空闲/回家，就接
        - 如果我已经在去救别人，那就看这个是不是更近的那个，是就换
        """
        if merchant is None:
            return
        if not getattr(merchant, "under_attack", False):
            return

        # 没有目标 → 直接接警
        if self.target is None:
            self.target = merchant
            self.state = "to_target"
            return

        # 有目标 → 只在新的更近时替换
        try:
            dist_new = euclid(self.pos_f, merchant.pos)
            dist_old = euclid(self.pos_f, self.target.pos)
            if dist_new < dist_old:
                self.target = merchant
                self.state = "to_target"
        except Exception:
            # 有时候 merchant.pos 还没写好，就算了
            pass

    # =========================================================
    # 每一步的逻辑
    # =========================================================
    def step(self):
        # 0) 没目标而且不是正在去救人 → 主动扫一遍有没有被打的商船
        if self.state != "to_target" and self.target is None:
            under_attacks = [
                m for m in getattr(self.model, "merchant_agents", [])
                if getattr(m, "under_attack", False)
            ]
            if under_attacks:
                # 选最近的
                self.target = min(under_attacks, key=lambda m: euclid(self.pos_f, m.pos))
                self.state = "to_target"

        # 1) 状态：去救人
        if self.state == "to_target" and self.target is not None:
            # 1.1 目标还在被打吗？
            if not getattr(self.target, "under_attack", False):
                # 被别人救掉了 / 攻击结束 → 回基地
                self.target = None
                self.state = "rtb"
                self._move_to_base()
                return

            # 1.2 商船现在在哪儿？（要追踪实时位置）
            target_pos = getattr(self.target, "pos", None)
            if target_pos is None:
                # 取不到位置 → 回基地
                self.target = None
                self.state = "rtb"
                self._move_to_base()
                return

            # 1.3 朝商船走
            self._move_towards(target_pos)

            # 1.4 判断是否已经拦截
            if euclid(self.pos_f, target_pos) <= self.intercept_radius:
                # armament=1 → 必定成功
                navy_id = self.name if self.name is not None else self.unique_id
                merchant_id = getattr(self.target, "name", self.target.unique_id)
                if hasattr(self.model, "events"):
                    self.model.events.append(("DISRUPT", navy_id, merchant_id))

                # 让商船脱离被攻击状态
                try:
                    self.target.under_attack = False
                except Exception:
                    pass

                # 清空目标 → 回基地
                self.target = None
                self.state = "rtb"
                self._move_to_base()
                return

            # 1.x 还在路上就结束这一步
            return

        # 2) 状态：回基地
        if self.state == "rtb":
            self._move_to_base()
            # 到家了就 idle
            if euclid(self.pos_f, self.base_pos) < 0.5:
                self.state = "idle"
            return

        # 3) 状态：idle，什么都不做，保持位置
        if self.state == "idle":
            # 也要同步一次到 space
            self.model.space.move_agent(self, self.pos_f)

    # =========================================================
    # 小工具：走路 & 回家
    # =========================================================
    def _move_towards(self, target_pos: Tuple[float, float]):
        """朝某个连续坐标走一步"""
        dist_per_step = self.speed * self.model.step_time   # km / step
        self.pos_f = step_move(self.pos_f, target_pos, dist_per_step)
        # 同步到 ContinuousSpace
        self.model.space.move_agent(self, self.pos_f)

    def _move_to_base(self):
        """回基地（左上角）"""
        self._move_towards(self.base_pos)

    def advance(self):
        # 给 SimultaneousActivation 用的占位
        pass
