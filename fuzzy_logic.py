"""
Method 7 — Fuzzy Logic Controller.
Rules with smooth thresholds (membership functions), defuzzified by weighted sum.
Fully interpretable.
"""

import numpy as np
from decision.base import DecisionMethod


def trapezoid(x, a, b, c, d):
    """Trapezoidal membership function: ramp up [a,b], plateau [b,c], ramp down [c,d]."""
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


class FuzzyLogic(DecisionMethod):
    name = "Fuzzy Logic"
    description = "Fuzzy controller with smooth thresholds and weighted defuzzification."
    learning_type = "rule-based (hand-crafted)"

    def decide(self, state):
        theta = state["theta"]
        v_batt = state["v_batt"]
        v_sc = state["v_sc"]

        # Memberships for theta
        cold       = trapezoid(theta, -50, -40, -22, -15)
        moderate   = trapezoid(theta, -22, -15, 30, 45)
        warm       = trapezoid(theta, 30, 45, 85, 100)

        # Memberships for V_batt
        low_batt    = trapezoid(v_batt, 2.6, 2.8, 3.0, 3.15)
        ok_batt     = trapezoid(v_batt, 3.0, 3.15, 3.7, 3.9)

        # Memberships for V_sc
        low_sc      = trapezoid(v_sc, 1.8, 2.0, 2.2, 2.5)
        ok_sc       = trapezoid(v_sc, 2.2, 2.5, 3.3, 3.5)

        # Fuzzy rules (each contributes to one action with the min of antecedents)
        # Rule 1: if cold → LOW_TEMP
        # Rule 2: if low_batt → DEEP_CONSERVE
        # Rule 3: if moderate AND ok_batt → NORMAL
        # Rule 4: if warm AND ok_batt → NORMAL
        # Rule 5: if low_sc AND cold → LOW_TEMP (boost SC)
        # Rule 6: if low_batt AND low_sc → DEEP_CONSERVE

        scores = [0.0, 0.0, 0.0]   # [NORMAL, LOW_TEMP, DEEP_CONSERVE]
        scores[1] += cold
        scores[2] += low_batt
        scores[0] += min(moderate, ok_batt)
        scores[0] += min(warm, ok_batt)
        scores[1] += min(low_sc, cold)
        scores[2] += min(low_batt, low_sc)

        return int(np.argmax(scores))

    def memory_bytes(self):
        # ~8 membership function parameters × 4 bytes + a few rules
        return 32 + 16
