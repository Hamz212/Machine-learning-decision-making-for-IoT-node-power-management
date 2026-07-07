"""
Method 7 — Fuzzy Logic Controller.
Rules with smooth thresholds (membership functions), defuzzified by weighted sum.
"""

import numpy as np
from decision.base import DecisionMethod


def trapezoid(x, a, b, c, d):
    """Trapezoidal membership function."""
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

        cold = trapezoid(theta, -50, -40, -22, -15)
        moderate = trapezoid(theta, -22, -15, 30, 45)
        warm = trapezoid(theta, 30, 45, 85, 100)

        low_batt = trapezoid(v_batt, 2.6, 2.8, 3.0, 3.15)
        ok_batt = trapezoid(v_batt, 3.0, 3.15, 3.7, 3.9)

        low_sc = trapezoid(v_sc, 1.8, 2.0, 2.2, 2.5)
        ok_sc = trapezoid(v_sc, 2.2, 2.5, 3.3, 3.5)

        scores = [0.0, 0.0, 0.0]
        scores[1] += cold
        scores[2] += low_batt
        scores[0] += min(moderate, ok_batt)
        scores[0] += min(warm, ok_batt)
        scores[1] += min(low_sc, cold)
        scores[2] += min(low_batt, low_sc)

        return int(np.argmax(scores))

    def memory_bytes(self):
        return 32 + 16
