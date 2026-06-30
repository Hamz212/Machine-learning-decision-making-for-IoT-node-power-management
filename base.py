"""
Base class — every decision method implements this interface.

The benchmark uses ONLY these two methods:
- train(env_factory)  : optional training phase (offline)
- decide(state)       : called at each wake-up to choose an action
"""

import time


class DecisionMethod:
    """Common interface for all decision methods."""

    name = "unnamed"
    description = "no description"
    learning_type = "unknown"   # supervised / unsupervised / RL / optimization / hybrid / rule-based

    def train(self, env_factory):
        """
        Optional training. Receives a callable returning a fresh environment.
        Default: no training needed.
        """
        pass

    def decide(self, state):
        """
        Return action (int): 0=NORMAL, 1=LOW_TEMP, 2=DEEP_CONSERVE.
        state is a dict: {theta, v_batt, v_sc, packets_pending}.
        """
        raise NotImplementedError

    def memory_bytes(self):
        """Return the memory footprint (octets) for deployment."""
        return 0
