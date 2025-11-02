import math, random
from typing import Tuple
from mesa import Model, Agent
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from MerchantShip import MerchantAgent


# --- Utility function ---
def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --- Minimal merchant placeholder ---
class MerchantAgent(Agent):
    """A minimal dummy merchant for testing PirateAgent."""
    def __init__(self, unique_id, model, speed_kn=12, alertness=0.5):
        super().__init__(unique_id, model)
        self.speed_kn = speed_kn
        self.alertness = alertness
        self.awareness = False

    def step(self):
        # keep stationary for simplicity
        pass

    def receive_distress(self, pirate_pos):
        print(f"🚨 Merchant {self.unique_id} distress at {self.pos}, pirate at {pirate_pos}")


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
        visibility_nm=10000,
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
        step = speed_kn * hours
        cur = self.pos
        dx, dy = dest[0] - cur[0], dest[1] - cur[1]
        d = math.hypot(dx, dy)
        if d <= step or d == 0:
            new_pos = dest
        else:
            new_pos = (cur[0] + dx / d * step, cur[1] + dy / d * step)
        self.model.space.move_agent(self, new_pos)

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
        if self.current_target_merchant is None:
            self.state = self.STATE_SEARCH
            return
        merchant_pos = self.current_target_merchant.pos
        self._move_towards(merchant_pos, self.pursuit_speed, hours)
        if distance(self.pos, merchant_pos) <= 0.2:
            merchant = self.current_target_merchant
            spotted = random.random() < min(1.0, merchant.alertness * hours)
            if spotted:
                merchant.awareness = True
                merchant.receive_distress(self.pos)
            self.state = self.STATE_ATTACK
            self.attack_timer = 0.0

    def _attack(self, hours):
        self.attack_timer += hours
        if self.attack_timer >= self.attack_time:
            merchant = self.current_target_merchant
            s = merchant.speed_kn
            m_base = 10.0
            pa = max(0.0, (2.0 - s / m_base) * self.qa)
            pu = max(0.0, (2.0 - s / m_base) * self.qu)
            prob = pa if merchant.awareness else pu
            if random.random() < prob:
                self.model.hijack_count += 1
                print(f"💀 Pirate {self.unique_id} hijacked {merchant.unique_id}!")
                try:
                    self.model.schedule.remove(merchant)
                except Exception:
                    pass
            self.state = self.STATE_RECUP
            self.cooldown_timer = 0.0

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


# --- Minimal test model ---
class PirateOnlyModel(Model):
    """A minimal Mesa 2.x model for testing PirateAgent."""
    def __init__(self, width=300, height=200, num_pirates=1, hours_per_step=1):
        super().__init__()
        self.space = ContinuousSpace(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.hours_per_step = hours_per_step
        self.hijack_count = 0
        self.attack_count = 0
        self.merchant_density_grid = {(100, 100): 5, (200, 120): 2}
        self.navy_positions = []

        # add one merchant
        merchant = MerchantAgent("merchant_1", self)
        self.space.place_agent(merchant, (100, 100))
        self.schedule.add(merchant)

        # add pirates
        for i in range(num_pirates):
            home = (random.uniform(10, width * 0.2), random.uniform(0, height))
            pirate = PirateAgent(f"pirate_{i}", self, home_anchor=home)
            self.space.place_agent(pirate, home)
            self.schedule.add(pirate)

    def step(self):
        self.schedule.step()


# --- Run a quick simulation ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = PirateOnlyModel(width=300, height=200, num_pirates=1)
    pirate = [a for a in model.schedule.agents if isinstance(a, PirateAgent)][0]

    pirate_positions = []
    for step in range(100):
        model.step()
        pirate_positions.append(pirate.pos)
        if step % 10 == 0:
            print(f"Step {step:03d}: state={pirate.state}, pos={pirate.pos}")

    print("\n✅ Simulation finished.")
    print(f"Final state: {pirate.state}")
    print(f"Hijack count: {model.hijack_count}")

    xs, ys = zip(*pirate_positions)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, 'r-', label="Pirate trajectory")
    plt.scatter(xs[0], ys[0], c='blue', label="Start (home)")
    plt.title("Pirate-only simulation (Mesa 2.x)")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    plt.show()
