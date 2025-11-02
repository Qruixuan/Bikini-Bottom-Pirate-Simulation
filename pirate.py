import math, random
from typing import Tuple
from mesa import Model, Agent
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from MerchantShip import MerchantAgent, distance


class PirateAgent(Agent):
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
            max_sailing_steps=100,
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
        self.max_sailing_steps = max_sailing_steps
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
        self.sailing_steps = 0

    def step(self):
        hours = self.model.hours_per_step

        # 只在攻击阶段检查是否遭遇海军，其余阶段忽略
        if self.state == self.STATE_ATTACK:
            if hasattr(self.model, "schedule") and self.pos is not None:
                for agent in self.model.schedule.agents:
                    if agent.__class__.__name__ == "NavyAgent" and agent.pos is not None:
                        dnavy = distance(self.pos, agent.pos)
                        if dnavy < self.visibility:
                            self._trigger_return(reason="navy_during_attack")
                            return

        # 航行步数计数与上限检查
        if self.state in (self.STATE_CRUISE, self.STATE_SEARCH, self.STATE_PURSUIT, self.STATE_ATTACK):
            self.sailing_steps += 1
            if self.sailing_steps >= self.max_sailing_steps:
                self._trigger_return(reason="max_sailing_steps")
                return

        # 状态机
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

    def _trigger_return(self, reason=None):
        self.current_target_merchant = None
        self.target_cell = self.home_anchor
        self.state = self.STATE_RETURN
        self.sailing_steps = 0
        if reason:
            print(f"→ Pirate {self.unique_id} triggered return due to: {reason}")

    def _select_target_area(self):
        anchor = getattr(self, "home_anchor", None)
        home_sigma = getattr(self, "home_sigma", 200.0)
        grid = getattr(self.model, "merchant_density_grid", None)

        if grid and len(grid) > 0:
            merged = {}
            for cell_pos, val in grid.items():
                weight = float(val)
                # 偏向 home_anchor
                if anchor is not None:
                    dx = cell_pos[0] - anchor[0]
                    dy = cell_pos[1] - anchor[1]
                    d = math.hypot(dx, dy)
                    sigma = max(1e-6, float(home_sigma))
                    home_factor = math.exp(- (d * d) / (2.0 * sigma * sigma))
                    weight *= home_factor
                merged[cell_pos] = max(weight, 0.0)

            total = sum(merged.values())
            if total <= 0:
                if anchor is not None:
                    nearest = min(grid.keys(), key=lambda p: math.hypot(p[0]-anchor[0], p[1]-anchor[1]))
                    self.target_cell = nearest
                else:
                    self.target_cell = random.choice(list(grid.keys()))
            else:
                r, cum = random.random() * total, 0.0
                for pos, val in merged.items():
                    cum += val
                    if r <= cum:
                        self.target_cell = pos
                        break
        else:
            if anchor is not None:
                sigma = max(1e-6, float(home_sigma))
                x = random.gauss(anchor[0], sigma)
                y = random.gauss(anchor[1], sigma)
                self.target_cell = (x, y)
            else:
                x = random.uniform(0, self.model.space.x_max)
                y = random.uniform(0, self.model.space.y_max)
                self.target_cell = (x, y)
        self.sailing_steps = 0
        self.state = self.STATE_CRUISE

    def _move_towards(self, dest, speed_kn, hours):
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
        x_max = getattr(self.model.space, 'x_max', 1000)
        y_max = getattr(self.model.space, 'y_max', 1000)
        clamped_x = max(0.0, min(new_pos[0], x_max - EPS))
        clamped_y = max(0.0, min(new_pos[1], y_max - EPS))
        self.model.space.move_agent(self, (clamped_x, clamped_y))

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
            max(0.0, min(cur[0] + jitter_x, self.model.space.x_max - EPS)),
            max(0.0, min(cur[1] + jitter_y, self.model.space.y_max - EPS)),
        )
        self.model.space.move_agent(self, new_pos)
        for agent in self.model.schedule.agents:
            if isinstance(agent, MerchantAgent):
                if distance(self.pos, agent.pos) <= self.visibility:
                    self.current_target_merchant = agent
                    self.state = self.STATE_PURSUIT
                    return
        self.time_since_departure += hours
        if self.time_since_departure >= self.max_sailing_steps:
            self.state = self.STATE_RETURN

    def _pursue(self, hours):
        if self.current_target_merchant is None:
            self.state = self.STATE_SEARCH
            return
        merchant = self.current_target_merchant
        if merchant.state == MerchantAgent.STATE_IN_PORT or merchant.pos is None:
            self.state = self.STATE_SEARCH
            self.current_target_merchant = None
            return
        self._move_towards(merchant.pos, self.pursuit_speed, hours)
        if distance(self.pos, merchant.pos) <= 0.2:
            if merchant.awareness or merchant.state == MerchantAgent.STATE_EVADING:
                merchant.awareness = True
                merchant.receive_distress(self.pos)
            self.state = self.STATE_ATTACK
            self.attack_timer = 0.0

    def _attack(self, hours):
        """攻击行为，若此时发现海军则立即撤退"""
        # 海军检查（仅在攻击阶段触发）
        if hasattr(self.model, "schedule") and self.pos is not None:
            for agent in self.model.schedule.agents:
                if agent.__class__.__name__ == "NavyAgent" and agent.pos is not None:
                    dnavy = distance(self.pos, agent.pos)
                    if dnavy < self.visibility:
                        print(f"⚓ Pirate {self.unique_id} spotted Navy during attack! Retreating!")
                        self._trigger_return(reason="navy_during_attack")
                        return

        self.attack_timer += hours
        if self.attack_timer >= self.attack_time:
            merchant = self.current_target_merchant
            if merchant not in self.model.schedule.agents:
                self.state = self.STATE_RECUP
                self.cooldown_timer = 0.0
                self.current_target_merchant = None
                return
            s = merchant.normal_speed
            m_base = 10.0
            pa = max(0.0, (2.0 - s / m_base) * self.qa)
            pu = max(0.0, (2.0 - s / m_base) * self.qu)
            prob = pa if merchant.awareness else pu
            if random.random() < prob:
                self.model.hijack_count += 1
                print(f"💀 Pirate {self.unique_id} hijacked {merchant.unique_id}!")
                self.model.schedule.remove(merchant)
            self.state = self.STATE_RECUP
            self.cooldown_timer = 0.0
            self.current_target_merchant = None

    def _recuperate(self, hours):
        self.cooldown_timer += hours
        if self.cooldown_timer >= self.cool_down:
            self.current_target_merchant = None
            if self.time_since_departure >= self.max_sailing_steps:
                self.state = self.STATE_RETURN
            else:
                self.state = self.STATE_SEARCH

    def _return_home(self, hours):
        self._move_towards(self.home_anchor, self.cruising_speed, hours)
        if distance(self.pos, self.home_anchor) < 1.0:
            self.time_since_departure = 0.0
            self.sailing_steps = 0
            self.state = self.STATE_SELECT