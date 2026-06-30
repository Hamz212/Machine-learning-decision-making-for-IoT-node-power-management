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
        # Initial thresholds (default = paper)
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
        """Genetic algorithm: evolve a population of threshold sets."""
        rng = np.random.default_rng(42)
        POP_SIZE = 20
        N_GEN = 15
        MUTATION_RATE = 0.3

        # Initialize population
        population = []
        for _ in range(POP_SIZE):
            individual = (
                rng.uniform(-30, -10),     # theta_l
                rng.uniform(2.9, 3.3),     # v_batt_l
                rng.uniform(2.0, 2.6),     # v_sc_l
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
            # Higher autonomy + transmissions, fewer failures
            score = (env.autonomy_days() * 100
                     + env.transmissions * 0.5
                     - env.failures * 1000)
            return score

        for gen in range(N_GEN):
            scored = [(fitness(ind), ind) for ind in population]
            scored.sort(reverse=True, key=lambda x: x[0])
            elites = [ind for _, ind in scored[:POP_SIZE // 4]]
            # Reproduce by crossover + mutation
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

        # Keep best
        best_score, best_ind = max(
            (fitness(ind), ind) for ind in population)
        self.theta_l, self.v_batt_l, self.v_sc_l = best_ind
