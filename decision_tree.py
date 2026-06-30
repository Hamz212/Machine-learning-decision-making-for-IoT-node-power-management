"""
Method 6 — Decision Tree learned from the Q-table oracle.
Fully interpretable: each decision is a cascade of comparisons.
"""

import numpy as np
from decision.base import DecisionMethod
from decision.q_learning import QLearning


class TreeNode:
    """Binary tree node."""
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreePolicy(DecisionMethod):
    name = "Decision Tree"
    description = "Decision tree trained by supervised distillation from Q-learning."
    learning_type = "supervised (from RL oracle)"

    MAX_DEPTH = 6

    def __init__(self):
        self.tree = None
        self._size = 0

    @staticmethod
    def _features(state):
        return np.array([state["theta"], state["v_batt"], state["v_sc"]],
                        dtype=np.float32)

    def decide(self, state):
        x = self._features(state)
        node = self.tree
        while node.value is None:
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return int(node.value)

    def memory_bytes(self):
        # Each non-leaf node: ~6 bytes (feature_id + threshold).
        # Each leaf: 1 byte.
        return self._size * 6

    def train(self, env_factory):
        # Build oracle dataset
        oracle = QLearning()
        oracle.train(env_factory)

        rng = np.random.default_rng(2)
        N = 20000
        X = np.zeros((N, 3), dtype=np.float32)
        y = np.zeros(N, dtype=np.int32)
        for i in range(N):
            st = {
                "theta": rng.uniform(-40, 85),
                "v_batt": rng.uniform(2.8, 3.7),
                "v_sc": rng.uniform(1.8, 3.3),
                "packets_pending": 0,
            }
            X[i] = self._features(st)
            y[i] = oracle.decide(st)

        # Build tree (CART, gini)
        self.tree = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        # Stop conditions
        if len(y) == 0:
            self._size += 1
            return TreeNode(value=0)
        majority = int(np.bincount(y, minlength=3).argmax())
        if depth >= self.MAX_DEPTH or len(np.unique(y)) == 1:
            self._size += 1
            return TreeNode(value=majority)

        best_gini = 1.0
        best_feat, best_thr = 0, 0.0
        best_split = None
        for f in range(3):
            sorted_vals = np.unique(X[:, f])
            if len(sorted_vals) < 2:
                continue
            # Try a few candidate splits
            for q in np.linspace(0.1, 0.9, 9):
                thr = np.quantile(X[:, f], q)
                left_mask = X[:, f] < thr
                if left_mask.sum() == 0 or left_mask.sum() == len(y):
                    continue
                gL = self._gini(y[left_mask])
                gR = self._gini(y[~left_mask])
                w = left_mask.sum() / len(y)
                g = w * gL + (1-w) * gR
                if g < best_gini:
                    best_gini = g
                    best_feat = f
                    best_thr = float(thr)
                    best_split = left_mask
        if best_split is None:
            self._size += 1
            return TreeNode(value=majority)

        left = self._build(X[best_split], y[best_split], depth+1)
        right = self._build(X[~best_split], y[~best_split], depth+1)
        self._size += 1
        return TreeNode(feature=best_feat, threshold=best_thr,
                        left=left, right=right)

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y, minlength=3) / len(y)
        return 1.0 - np.sum(p ** 2)
