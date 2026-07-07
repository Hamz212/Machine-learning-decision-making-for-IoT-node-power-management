"""
Method 2 — FSM with thresholds optimized via genetic algorithm.
Keeps the FSM structure, but learns optimal thresholds.
"""

import numpy as np
from decision.base import DecisionMethod


class FSMOptimized(DecisionMethod):
    name = "FSM Optimized (GA)"
    description = "FSM with thresholds optimized by a simple genetic algorithm."
    learning_type = "optimization"

    def __init__(self):
        self.theta_l = -20.0
        self.v_batt_l = 3.0
        self.v_sc_l = 2.2

    def decide(self, state):
        if state["v_batt"] < self.v_batt_l: return 2
        if state["theta"] < self.theta_l: return 1
        return 0

    def memory_bytes(self):
        return 6

    def train(self, env_factory):
        rng = np.random.default_rng(42)
        POP_SIZE = 10       # reduced from 20 -- each fitness is now 60 days
        N_GEN = 8           # reduced from 15
        MUTATION_RATE = 0.3

        population = []
        for _ in range(POP_SIZE):
            individual = (
                rng.uniform(-30, -10),
                rng.uniform(2.9, 3.3),
                rng.uniform(2.0, 2.6),
            )
            population.append(individual)

        def fitness(individual):
            self.theta_l, self.v_batt_l, self.v_sc_l = individual
            env = env_factory()
            done = False
            while not done:
                state = env.get_state()
                action = self.decide(state)
                state, done = env.step(action)
            auton = env.autonomy_days()
            # SERVICE MINIMUM : reject solutions that don't transmit
            # (at least 10 TX per day of autonomy)
            tx_per_day = env.transmissions / max(auton, 0.01)
            if tx_per_day < 10:
                return -1e6   # unacceptable: not serving purpose
            # Otherwise : maximize autonomy, minimize energy per day
            # (energy_per_day is a proxy for how efficient the strategy is)
            energy_per_day = env.total_energy_consumed / max(auton, 0.01)
            score = (auton * 1000                       # autonomy dominant
                     - energy_per_day * 10               # penalize gluttony
                     + env.transmissions * 0.05          # small bonus for TX
                     - env.failures * 100000)
            return score

        for gen in range(N_GEN):
            scored = [(fitness(ind), ind) for ind in population]
            scored.sort(reverse=True, key=lambda x: x[0])
            elites = [ind for _, ind in scored[:POP_SIZE // 4]]
            new_pop = list(elites)
            while len(new_pop) < POP_SIZE:
                p1, p2 = rng.choice(len(elites), 2, replace=True)
                child = []
                for a, b in zip(elites[p1], elites[p2]):
                    val = (a + b) / 2
                    if rng.random() < MUTATION_RATE:
                        val += rng.normal(0, 0.3)
                    child.append(val)
                new_pop.append(tuple(child))
            population = new_pop

        best_score, best_ind = max(
            (fitness(ind), ind) for ind in population)
        self.theta_l, self.v_batt_l, self.v_sc_l = best_ind
