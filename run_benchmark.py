"""
Benchmark of decision methods for the IoT node power manager.

Runs each method on the same standard TVCC scenario, with the same seed,
and reports a comparison table.
"""

import time
import sys
import os
import numpy as np

# Make sure we can import the local packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import IoTNodeEnv
from decision.fsm_baseline import FSMBaseline
from decision.fsm_optimized import FSMOptimized
from decision.random_policy import RandomPolicy
from decision.q_learning import QLearning
from decision.mlp_distilled import MLPDistilled
from decision.decision_tree import DecisionTreePolicy
from decision.fuzzy_logic import FuzzyLogic


# Fixed scenario: 1 week of standard TVCC, seed 42
EPISODE_MINUTES = 60 * 24 * 7   # 7 days
SCENARIO_SEED = 42


def make_env_fresh(seed=None):
    """Factory that returns a fresh env with the standard scenario."""
    return IoTNodeEnv(seed=seed if seed is not None else SCENARIO_SEED,
                      episode_minutes=EPISODE_MINUTES)


def run_episode(method, env):
    """Run one episode with a given method."""
    state = env.get_state()
    done = False
    decision_times = []
    while not done:
        t0 = time.perf_counter()
        a = method.decide(state)
        decision_times.append(time.perf_counter() - t0)
        state, done = env.step(a)
    return decision_times


def evaluate(method, n_eval_seeds=5):
    """Average across multiple eval seeds for robustness."""
    results = {
        "autonomy_days": [],
        "failures": [],
        "transmissions": [],
        "dropped_packets": [],
        "energy_J": [],
        "decision_us_avg": [],
        "tx_interval_avg": [],
    }
    for s in range(n_eval_seeds):
        env = IoTNodeEnv(seed=1000 + s, episode_minutes=EPISODE_MINUTES)
        decision_times = run_episode(method, env)
        results["autonomy_days"].append(env.autonomy_days())
        results["failures"].append(env.failures)
        results["transmissions"].append(env.transmissions)
        results["dropped_packets"].append(env.dropped_packets)
        results["energy_J"].append(env.total_energy_consumed)
        results["decision_us_avg"].append(np.mean(decision_times) * 1e6)
        if env.tx_intervals:
            results["tx_interval_avg"].append(np.mean(env.tx_intervals))
        else:
            results["tx_interval_avg"].append(float("nan"))
    # Aggregate
    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    methods = [
        RandomPolicy(seed=7),
        FSMBaseline(),
        FSMOptimized(),
        FuzzyLogic(),
        QLearning(),
        DecisionTreePolicy(),
        MLPDistilled(),
    ]

    print("=" * 85)
    print(f"BENCHMARK — {len(methods)} methods, standard TVCC scenario, "
          f"1 week × 5 eval seeds")
    print("=" * 85)

    results = []
    for m in methods:
        print(f"\n▶ {m.name} ({m.learning_type})")
        # Train if applicable
        t0 = time.perf_counter()
        m.train(env_factory=make_env_fresh)
        train_time = time.perf_counter() - t0
        # Evaluate
        metrics = evaluate(m, n_eval_seeds=5)
        metrics["method"] = m.name
        metrics["learning_type"] = m.learning_type
        metrics["train_s"] = train_time
        metrics["memory_B"] = m.memory_bytes()
        results.append(metrics)
        print(f"  autonomy={metrics['autonomy_days']:.2f}d  "
              f"failures={metrics['failures']:.1f}  "
              f"TX={metrics['transmissions']:.0f}  "
              f"decide={metrics['decision_us_avg']:.1f}µs  "
              f"mem={metrics['memory_B']}B  "
              f"train={train_time:.2f}s")

    # Final table
    print("\n" + "=" * 110)
    print("RESULTS SUMMARY")
    print("=" * 110)
    headers = [
        ("method", 22, "s"),
        ("learning_type", 22, "s"),
        ("autonomy_days", 8, ".2f"),
        ("failures", 8, ".1f"),
        ("transmissions", 8, ".0f"),
        ("dropped_packets", 7, ".0f"),
        ("decision_us_avg", 10, ".1f"),
        ("memory_B", 8, "d"),
        ("train_s", 8, ".2f"),
    ]
    short = {
        "method": "Method",
        "learning_type": "Learning",
        "autonomy_days": "Auton(d)",
        "failures": "Fails",
        "transmissions": "TX",
        "dropped_packets": "Drops",
        "decision_us_avg": "T_dec(µs)",
        "memory_B": "Mem(B)",
        "train_s": "Train(s)",
    }
    # Header line
    line = ""
    for col, w, _ in headers:
        line += f"{short[col]:<{w}} "
    print(line)
    print("-" * 110)
    for r in results:
        line = ""
        for col, w, fmt in headers:
            val = r[col]
            if fmt == "s":
                s = f"{str(val)[:w]:<{w}}"
            elif fmt == "d":
                s = f"{int(val):<{w}d}"
            else:
                s = f"{val:<{w}{fmt}}"
            line += s + " "
        print(line)
    print("=" * 110)

    # Save CSV
    os.makedirs("results", exist_ok=True)
    import csv
    csv_path = os.path.join("results", "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[c[0] for c in headers])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k, _, _ in headers})
    print(f"\n✓ Results saved to {csv_path}")

    return results


if __name__ == "__main__":
    main()
