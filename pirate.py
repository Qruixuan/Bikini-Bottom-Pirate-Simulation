import math, random
from typing import Tuple
from mesa import Model, Agent
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
# 导入 MerchantAgent 类和 distance 函数
from MerchantShip import MerchantAgent, distance


# --- Pirate Agent ---
class PirateAgent(Agent):
    """Pirate behavior cycle: select → cruise → search → pursuit/attack → recuperate → return"""
    STATE_SELECT = "select"
    STATE_CRUISE = "cruise"
    STATE_SEARCH = "search"
    STATE_PURSUIT = "pursuit"
    STATE_ATTACK = "attack"
    STATE_RECUP = "recuperate"
    STATE_RETURN = "return_home"

    def __init__(
            self, unique_id, model,
            home_anchor=(0, 0),
            cruising_speed_kn=10,
            pursuit_speed_kn=28,
            endurance_days=14,
            visibility_nm=80,
            attack_time_hrs=0.5,
            cool_down_hrs=2,
            navy_knowledge_prob=0.4,
            qa=0.2, qu=0.5
    ):
        super().__init__(unique_id, model)
        self.home_anchor = home_anchor
        self.cruising_speed = cruising_speed_kn
        self.pursuit_speed = pursuit_speed_kn
        self.endurance = endurance_days * 24.0
        self.visibility = visibility_nm
        self.attack_time = attack_time_hrs
        self.cool_down = cool_down_hrs
        self.navy_knowledge = navy_knowledge_prob
        self.qa = qa
        self.qu = qu

        self.state = PirateAgent.STATE_SELECT
        self.time_since_departure = 0.0
        self.target_cell = None
        self.search_time = 0.0
        self.current_target_merchant = None
        self.cooldown_timer = 0.0

    # --- Main step ---
    def step(self):
        hours = self.model.hours_per_step
        if self.state == self.STATE_SELECT:
            self._select_target_area()
        elif self.state == self.STATE_CRUISE:
            self._cruise(hours)
        elif self.state == self.STATE_SEARCH:
            self._search(hours)
        elif self.state == self.STATE_PURSUIT:
            self._pursue(hours)
        elif self.state == self.STATE_ATTACK:
            self._attack(hours)
        elif self.state == self.STATE_RECUP:
            self._recuperate(hours)
        elif self.state == self.STATE_RETURN:
            self._return_home(hours)

    # --- Internal behaviors ---
    def _select_target_area(self):
        grid = getattr(self.model, "merchant_density_grid", None)
        if grid and len(grid) > 0:
            merged = {}
            for cell_pos, val in grid.items():
                weight = val
                if (random.random() < self.navy_knowledge) and hasattr(self.model, "navy_positions"):
                    for npos in self.model.navy_positions:
                        d = distance(cell_pos, npos)
                        if d < 200:
                            weight *= 0.5
                merged[cell_pos] = max(weight, 0.0)
            total = sum(merged.values())
            if total <= 0:
                self.target_cell = random.choice(list(grid.keys()))
            else:
                r, cum = random.random() * total, 0.0
                for pos, val in merged.items():
                    cum += val
                    if r <= cum:
                        self.target_cell = pos
                        break
        else:
            x = random.uniform(0, self.model.space.x_max)
            y = random.uniform(0, self.model.space.y_max)
            self.target_cell = (x, y)
        self.state = self.STATE_CRUISE

    def _move_towards(self, dest, speed_kn, hours):
        """
        移动逻辑（已修正边界限制）。
        此方法将钳制新位置，确保它不会超出 ContinuousSpace 的边界。
        """
        # 增加防御性检查：如果当前位置或目标位置为 None，则不移动
        if self.pos is None or dest is None:
            return

        step = speed_kn * hours
        cur = self.pos
        dx, dy = dest[0] - cur[0], dest[1] - cur[1]
        d = math.hypot(dx, dy)

        if d <= step or d == 0:
            new_pos = dest
        else:
            new_pos = (cur[0] + dx / d * step, cur[1] + dy / d * step)

        # --- 边界钳制/限制 ---
        x_max = getattr(self.model.space, 'x_max', 1000)
        y_max = getattr(self.model.space, 'y_max', 1000)

        # 钳制新位置，确保它不会超出 [0, max] 范围
        clamped_x = max(0.0, min(new_pos[0], x_max))
        clamped_y = max(0.0, min(new_pos[1], y_max))
        final_pos = (clamped_x, clamped_y)
        # --------------------------------

        self.model.space.move_agent(self, final_pos)

    def _cruise(self, hours):
        if self.target_cell is None:
            self.state = self.STATE_SELECT
            return
        self._move_towards(self.target_cell, self.cruising_speed, hours)
        if distance(self.pos, self.target_cell) < 1.0:
            self.state = self.STATE_SEARCH
            self.search_time = 0.0

    def _search(self, hours):
        self.search_time += hours
        cur = self.pos
        jitter_x, jitter_y = random.uniform(-1, 1), random.uniform(-1, 1)
        new_pos = (
            max(0, min(self.model.space.x_max, cur[0] + jitter_x)),
            max(0, min(self.model.space.y_max, cur[1] + jitter_y)),
        )
        self.model.space.move_agent(self, new_pos)

        # look for merchants
        for agent in self.model.schedule.agents:
            if isinstance(agent, MerchantAgent):
                if distance(self.pos, agent.pos) <= self.visibility:
                    self.current_target_merchant = agent
                    self.state = self.STATE_PURSUIT
                    return

        self.time_since_departure += hours
        if self.time_since_departure >= self.endurance:
            self.state = self.STATE_RETURN

    def _pursue(self, hours):
        """追击商船，如果发现海军则逃跑"""
        if self.current_target_merchant is None:
            self.state = self.STATE_SEARCH
            return

        merchant = self.current_target_merchant

        # 如果商船在港口则放弃追击
        if merchant.state == MerchantAgent.STATE_IN_PORT:
            self.state = self.STATE_SEARCH
            self.current_target_merchant = None
            return

        merchant_pos = merchant.pos
        if merchant_pos is None:
            self.state = self.STATE_SEARCH
            self.current_target_merchant = None
            return

        # ✅ 检查是否有海军在视野内
        nearest_navy = None
        nearest_navy_dist = float("inf")
        for agent in self.model.schedule.agents:
            # 用类名判断，不强依赖 NavyAgent 定义
            if agent.__class__.__name__ == "NavyAgent":
                if agent.pos is None:
                    continue
                d = distance(self.pos, agent.pos)
                if d < self.visibility and d < nearest_navy_dist:
                    nearest_navy_dist = d
                    nearest_navy = agent

        if nearest_navy:
            print(f"⚓ Pirate {self.unique_id} spotted navy at {nearest_navy_dist:.1f} nm → retreating!")
            # 立刻中止追击，返回基地
            self.current_target_merchant = None
            self.state = self.STATE_RETURN
            return

        # 如果安全，继续追击
        self._move_towards(merchant_pos, self.pursuit_speed, hours)

        # 如果到达目标附近，准备攻击
        if distance(self.pos, merchant_pos) <= 0.2:
            if merchant.awareness or merchant.state == MerchantAgent.STATE_EVADING:
                merchant.awareness = True
                merchant.receive_distress(self.pos)
            self.state = self.STATE_ATTACK
            self.attack_timer = 0.0

    def _attack(self, hours):
        self.attack_timer += hours
        if self.attack_timer >= self.attack_time:
            merchant = self.current_target_merchant

            if merchant not in self.model.schedule.agents:
                self.state = self.STATE_RECUP
                self.cooldown_timer = 0.0
                self.current_target_merchant = None  # 清除引用
                return

            s = merchant.normal_speed
            m_base = 10.0
            pa = max(0.0, (2.0 - s / m_base) * self.qa)
            pu = max(0.0, (2.0 - s / m_base) * self.qu)

            prob = pa if merchant.awareness else pu

            if random.random() < prob:
                self.model.hijack_count += 1
                print(f"💀 Pirate {self.unique_id} hijacked {merchant.unique_id}!")
                try:
                    self.model.schedule.remove(merchant)
                    merchant.pos = None

                    self.state = self.STATE_RECUP
                    self.cooldown_timer = 0.0
                    self.current_target_merchant = None  # 劫持成功，清除引用
                    return
                except Exception:
                    pass

            # 无论劫持成功与否，战斗结束后清除目标并进入恢复状态
            self.state = self.STATE_RECUP
            self.cooldown_timer = 0.0
            self.current_target_merchant = None  # 战斗结束，清除引用

    def _recuperate(self, hours):
        self.cooldown_timer += hours
        if self.cooldown_timer >= self.cool_down:
            self.current_target_merchant = None
            if self.time_since_departure >= self.endurance:
                self.state = self.STATE_RETURN
            else:
                self.state = self.STATE_SEARCH

    def _return_home(self, hours):
        self._move_towards(self.home_anchor, self.cruising_speed, hours)
        if distance(self.pos, self.home_anchor) < 1.0:
            self.time_since_departure = 0.0
            self.state = self.STATE_SELECT