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
        # 36 states × 3 actions × 1 byte (int8 quantized)
        return self.N_STATES * self.N_ACTIONS

    @staticmethod
    def _reward(env, prev_failures, prev_tx, action, theta):
        r = 0.1
        if env.transmissions > prev_tx: r += 10
        if env.failures > prev_failures: r -= 100
        if env.v_batt < 3.0: r -= 5
        if theta < -20 and action == 0: r -= 3   # NORMAL in extreme cold
        if theta > 0 and action == 2: r -= 2     # DEEP_CONSERVE when warm
        return r

    def train(self, env_factory):
        alpha = 0.1
        gamma = 0.95
        eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995
        N_EPISODES = 200
        eps = eps_start
        rng = np.random.default_rng(0)

        for ep in range(N_EPISODES):
            env = env_factory()
            state = env.get_state()
            s_idx = self._encode_state(state)
            done = False
            prev_fail = env.failures
            prev_tx = env.transmissions

            while not done:
                if rng.random() < eps:
                    a = rng.integers(0, self.N_ACTIONS)
                else:
                    a = int(np.argmax(self.Q[s_idx]))

                state_next, done = env.step(a)
                r = self._reward(env, prev_fail, prev_tx, a, state["theta"])
                prev_fail, prev_tx = env.failures, env.transmissions

                s_next = self._encode_state(state_next)
                target = r if done else r + gamma * np.max(self.Q[s_next])
                self.Q[s_idx, a] += alpha * (target - self.Q[s_idx, a])
                s_idx = s_next
                state = state_next

            eps = max(eps_end, eps * eps_decay)
