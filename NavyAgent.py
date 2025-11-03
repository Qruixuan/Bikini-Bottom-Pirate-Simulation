import math
import random
from typing import Tuple, List, Dict
import numpy as np
from MerchantShip import MerchantAgent, distance
from pirate import PirateAgent

import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
import matplotlib.animation as animation

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

class NavyAgent(Agent):
    """
    连续空间海军（带燃料/航行步数版）：
    - 听到商船呼救马上出发
    - 任务结束一定回基地
    - 回基地途中如果还有油，可以被新的求救打断，再次前往
    - 每艘船有 max_steps 的最大航行步数，耗尽后必须先回基地加油
    """
    def __init__(self, unique_id, model,
                 speed: float = 40.0,
                 armament: float = 1.0,
                 base_pos: Tuple[float, float] | None = None,
                 max_steps: int = 200):
        super().__init__(unique_id, model)
        self.speed = speed
        self.armament = armament
        self.intercept_radius = 1.0
        self.pos_f = (0.0, 0.0)
        self.target = None
        if base_pos is None:
            base_pos = (0.0, self.model.space.y_max)
        self.base_pos = base_pos

        # 状态: idle / to_target / rtb
        self.state = "idle"

        # 燃料/步数
        self.max_steps = max_steps       # 最大能走多少步
        self.steps_left = max_steps      # 当前还能走多少步

    # ------------------ 对外接口：接收求救 ------------------
    def receive_distress(self, merchant: MerchantAgent):
        # 没油了 → 直接忽略这次呼叫
        if self.steps_left <= 0:
            return

        if merchant is None or merchant.pos is None:
            return
        if not getattr(merchant, "under_attack", False):
            return

        # 如果现在没有任务，直接接
        if self.target is None:
            self.target = merchant
            self.state = "to_target"
            return

        # 有任务 → 换成离自己更近的那个
        try:
            old_dist = distance(self.pos_f, self.target.pos)
            new_dist = distance(self.pos_f, merchant.pos)
            if new_dist < old_dist:
                self.target = merchant
                self.state = "to_target"
        except Exception:
            pass

    def can_accept_mission(self, pos: tuple[float, float]) -> bool:
        """
        判断当前海军是否可以接受新的任务。
        参数:
            pos: 任务目标的位置 (x, y)
        返回:
            True  -> 可以接受任务
            False -> 暂时无法接受
        """
        # 如果当前状态是待命 → 可以直接接
        if self.state == "idle":
            return True

        # 如果当前状态是执行任务中 → 不可接
        if self.state == "to_target":
            return False

        # 如果正在返航
        if self.state == "rtb":
            # 计算距离基地的距离
            dist_to_base = distance(self.pos_f, self.base_pos)
            # 估算返航油量阈值（剩余步数必须 > 往返消耗）
            if self.steps_left > dist_to_base / (self.speed * self.model.hours_per_step):
                return True  # 还有油，能接任务
            else:
                return False  # 油不够，必须先回去加油

        # 其他未知状态默认不接任务
        return False

    # ------------------ 主逻辑 ------------------
    def step(self):
        hours = self.model.hours_per_step
        dist_per_step = self.speed * hours

        # ① 没油了 → 不跟你讲道理，直接回去
        if self.steps_left <= 0 and self.state != "rtb":
            self.target = None
            self.state = "rtb"

        # ② 如果空闲但外面有人在挨打 → 主动出动（前提是有油）
        if self.state == "idle" and self.steps_left > 0:
            ua = [m for m in self.model.merchant_agents if getattr(m, "under_attack", False)]
            if ua:
                self.target = min(ua, key=lambda m: distance(self.pos_f, m.pos))
                self.state = "to_target"

        # ③ 去救人
        if self.state == "to_target" and self.target is not None:
            # 还没走就发现对方不挨打了 → 回去
            if not getattr(self.target, "under_attack", False):
                self.target = None
                self.state = "rtb"
                self._move_to_base(dist_per_step)
                return

            target_pos = getattr(self.target, "pos", None)
            if target_pos is None:
                # 商船可能被劫持/被删了
                self.target = None
                self.state = "rtb"
                self._move_to_base(dist_per_step)
                return

            # 真正去追
            self.pos_f = step_move(self.pos_f, target_pos, dist_per_step)
            self.model.space.move_agent(self, self.pos_f)
            self.steps_left -= 1   # 走一步扣一步

            # 到达目标附近 → 任务结束，强制回家
            if distance(self.pos_f, target_pos) <= self.intercept_radius:
                # 拦截成功就清掉标志
                if getattr(self.target, "under_attack", False):
                    self.target.under_attack = False
                if hasattr(self.model, "events"):
                    self.model.events.append(("DISRUPT", self.unique_id, self.target.unique_id))
                self.target = None
                self.state = "rtb"
            return

        # ④ 回基地
        if self.state == "rtb":
            # 返程途中如果还有油 → 可以被新的求救打断，前提是 steps_left > 0
            if self.steps_left > 0:
                ua = [m for m in self.model.merchant_agents if getattr(m, "under_attack", False)]
                if ua:
                    # 按你说的：途中又被呼叫 → 再去
                    closest = min(ua, key=lambda m: distance(self.pos_f, m.pos))
                    self.target = closest
                    self.state = "to_target"
                    return

            # 正常返航
            self._move_to_base(dist_per_step)
            self.steps_left -= 1   # 返航也要扣油

            # 到家 → 补满油，变 idle
            if distance(self.pos_f, self.base_pos) < 0.5:
                self.steps_left = self.max_steps
                self.state = "idle"
                self.target = None
            return

        # ⑤ idle 原地耗不耗油？不耗
        if self.state == "idle":
            self.model.space.move_agent(self, self.pos_f)

    # ------------------ 辅助 ------------------
    def _move_to_base(self, dist):
        self.pos_f = step_move(self.pos_f, self.base_pos, dist)
        self.model.space.move_agent(self, self.pos_f)