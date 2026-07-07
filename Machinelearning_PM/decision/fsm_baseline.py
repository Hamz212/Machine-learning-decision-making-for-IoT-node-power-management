"""
Method 1 — FSM baseline (original from the I2MTC 2026 paper).
Fixed thresholds, no learning.
"""

from decision.base import DecisionMethod


class FSMBaseline(DecisionMethod):
    name = "FSM Baseline"
    description = "Original FSM from the I2MTC paper, with fixed thresholds."
    learning_type = "rule-based"

    THETA_L = -20.0
    V_BATT_L = 3.0
    V_SC_L = 2.2

    def decide(self, state):
        theta = state["theta"]
        v_batt = state["v_batt"]
        v_sc = state["v_sc"]

        if v_batt < self.V_BATT_L:
            return 2  # DEEP_CONSERVE
        if theta < self.THETA_L:
            return 1  # LOW_TEMP
        return 0  # NORMAL

    def memory_bytes(self):
        return 6  # 3 thresholds × 2 bytes
