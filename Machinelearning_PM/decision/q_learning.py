"""
Method 4 — Q-learning tabular.
Learns a discrete Q-table by interacting with the simulator.
"""

import numpy as np
from decision.base import DecisionMethod


class QLearning(DecisionMethod):
    name = "Q-learning tabulaire"
    description = "Reinforcement learning with a discrete Q-table."
    learning_type = "reinforcement"

    N_STATES = 4 * 3 * 3   # theta × v_batt × v_sc
    N_ACTIONS = 3

    def __init__(self):
        rng = np.random.default_rng(42)
        self.Q = rng.uniform(-0.01, 0.01, (self.N_STATES, self.N_ACTIONS))

    @staticmethod
    def _bin_theta(t):
        if t < -20: return 0
        elif t < 0: return 1
        elif t < 40: return 2
        else: return 3

    @staticmethod
    def _bin_vbatt(v):
        if v < 3.2: return 0
        elif v < 3.5: return 1
        else: return 2

    @staticmethod
    def _bin_vsc(v):
        if v < 2.2: return 0
        elif v < 2.9: return 1
        else: return 2

    def _encode_state(self, state):
        a = self._bin_theta(state["theta"])
        b = self._bin_vbatt(state["v_batt"])
        c = self._bin_vsc(state["v_sc"])
        return (a * 3 + b) * 3 + c

    def decide(self, state):
        idx = self._encode_state(state)
        return int(np.argmax(self.Q[idx]))

    def memory_bytes(self):
        return self.N_STATES * self.N_ACTIONS

    @staticmethod
    def _reward(env, prev_failures, prev_tx, action, theta, prev_energy):
        """
        Reward function : autonomy PRIMARY, but service required.
        """
        r = 0.0

        # Big penalty for consuming energy (batterie = ressource rare)
        energy_consumed = env.total_energy_consumed - prev_energy
        r -= energy_consumed * 200    # significant but not overwhelming

        # Survival bonus per minute
        r += 0.05

        # Failure = catastrophe
        if env.failures > prev_failures:
            r -= 5000

        # Low battery warning zone
        if env.v_batt < 3.1:
            r -= 1.0
        if env.v_batt < 2.9:
            r -= 3.0

        # Meaningful bonus for TX (this is the useful service)
        if env.transmissions > prev_tx:
            r += 3.0

        return r

    def train(self, env_factory):
        alpha = 0.1
        gamma = 0.95
        eps_start, eps_end, eps_decay = 1.0, 0.05, 0.98
        N_EPISODES = 30    # reduced from 200 -- each episode is now 60 days
        eps = eps_start
        rng = np.random.default_rng(0)

        for ep in range(N_EPISODES):
            env = env_factory()
            state = env.get_state()
            s_idx = self._encode_state(state)
            done = False
            prev_fail = env.failures
            prev_tx = env.transmissions
            prev_energy = env.total_energy_consumed

            while not done:
                if rng.random() < eps:
                    a = rng.integers(0, self.N_ACTIONS)
                else:
                    a = int(np.argmax(self.Q[s_idx]))

                state_next, done = env.step(a)
                r = self._reward(env, prev_fail, prev_tx, a,
                                 state["theta"], prev_energy)
                prev_fail = env.failures
                prev_tx = env.transmissions
                prev_energy = env.total_energy_consumed

                s_next = self._encode_state(state_next)
                target = r if done else r + gamma * np.max(self.Q[s_next])
                self.Q[s_idx, a] += alpha * (target - self.Q[s_idx, a])
                s_idx = s_next
                state = state_next

            eps = max(eps_end, eps * eps_decay)
