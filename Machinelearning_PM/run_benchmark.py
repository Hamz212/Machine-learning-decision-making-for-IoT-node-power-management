"""
Benchmark v2 -- OBJECTIF AUTONOMIE MAXIMALE.

Changements vs v1 :
- Episode etendu a 60 jours (au lieu de 7) pour stresser la batterie
- Metriques centrees sur l'autonomie et l'energie, pas les TX
- Recompense Q-learning et fitness GA revues pour prioriser la survie

Usage :
    python3 run_benchmark.py
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import time
import numpy as np

from environment import IoTNodeEnv
from decision.fsm_baseline import FSMBaseline
from decision.fsm_optimized import FSMOptimized
from decision.random_policy import RandomPolicy
from decision.q_learning import QLearning
from decision.mlp_distilled import MLPDistilled
from decision.decision_tree import DecisionTreePolicy
from decision.fuzzy_logic import FuzzyLogic


# === Configuration ===
EPISODE_MINUTES = 60 * 24 * 60   # 60 DAYS -- stress the battery
SCENARIO_SEED = 42


def make_env_fresh(seed=None):
    return IoTNodeEnv(seed=seed if seed is not None else SCENARIO_SEED,
                      episode_minutes=EPISODE_MINUTES)


def run_episode(method, env):
    state = env.get_state()
    done = False
    decision_times = []
    while not done:
        t0 = time.perf_counter()
        a = method.decide(state)
        decision_times.append(time.perf_counter() - t0)
        state, done = env.step(a)
    return decision_times


def evaluate(method, n_eval_seeds=3):
    """Run each method on multiple seeds and average."""
    results = {
        "autonomy_days": [],
        "failures": [],
        "transmissions": [],
        "dropped_packets": [],
        "energy_J": [],
        "energy_per_day_J": [],
        "tx_per_mJ": [],
        "useful_packets": [],
        "decision_us_avg": [],
    }
    for s in range(n_eval_seeds):
        env = IoTNodeEnv(seed=1000 + s, episode_minutes=EPISODE_MINUTES)
        decision_times = run_episode(method, env)

        auton = env.autonomy_days()
        results["autonomy_days"].append(auton)
        results["failures"].append(env.failures)
        results["transmissions"].append(env.transmissions)
        results["dropped_packets"].append(env.dropped_packets)
        results["energy_J"].append(env.total_energy_consumed)

        # Energy per day (efficiency): consumption normalized by survival time
        if auton > 0:
            results["energy_per_day_J"].append(env.total_energy_consumed / auton)
        else:
            results["energy_per_day_J"].append(float("nan"))

        # TX per J consumed (transmission efficiency)
        if env.total_energy_consumed > 0:
            results["tx_per_mJ"].append(
                env.transmissions / env.total_energy_consumed)
        else:
            results["tx_per_mJ"].append(0)

        # Useful packets = measured minus dropped (approximation)
        useful = env.transmissions * 4  # each TX empties ~4 packets in NORMAL mode
        results["useful_packets"].append(useful)

        results["decision_us_avg"].append(np.mean(decision_times) * 1e6)

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

    print("=" * 90)
    print(f"BENCHMARK v2 -- {len(methods)} methods, 60-day TVCC scenario, "
          f"3 eval seeds")
    print(f"OBJECTIF PRIORITAIRE : AUTONOMIE MAXIMALE")
    print("=" * 90)

    results = []
    for m in methods:
        print(f"\n> {m.name} ({m.learning_type})")
        t0 = time.perf_counter()
        m.train(env_factory=make_env_fresh)
        train_time = time.perf_counter() - t0
        metrics = evaluate(m, n_eval_seeds=3)
        metrics["method"] = m.name
        metrics["learning_type"] = m.learning_type
        metrics["train_s"] = train_time
        metrics["memory_B"] = m.memory_bytes()
        results.append(metrics)
        print(f"  autonomy={metrics['autonomy_days']:.1f}d  "
              f"E/day={metrics['energy_per_day_J']:.2f}J/d  "
              f"TX={metrics['transmissions']:.0f}  "
              f"TX/J={metrics['tx_per_mJ']:.2f}  "
              f"train={train_time:.1f}s")

    # Sort by autonomy (descending) for clarity
    results.sort(key=lambda r: -r["autonomy_days"])

    # Final table
    print("\n" + "=" * 130)
    print("RESULTS SUMMARY (sorted by autonomy)")
    print("=" * 130)
    headers = [
        ("method",              22, "s"),
        ("learning_type",       25, "s"),
        ("autonomy_days",       9,  ".2f"),
        ("energy_per_day_J",    10, ".3f"),
        ("transmissions",       9,  ".0f"),
        ("tx_per_mJ",           9,  ".2f"),
        ("dropped_packets",     8,  ".0f"),
        ("memory_B",            7,  "d"),
    ]
    short = {
        "method": "Method",
        "learning_type": "Learning type",
        "autonomy_days": "Auton(d)",
        "energy_per_day_J": "E/day(J)",
        "transmissions": "TX",
        "tx_per_mJ": "TX/J",
        "dropped_packets": "Drops",
        "memory_B": "Mem(B)",
    }
    line = ""
    for col, w, _ in headers:
        line += f"{short[col]:<{w}} "
    print(line)
    print("-" * 130)
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
    print("=" * 130)

    # Save CSV
    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
    import csv
    csv_path = os.path.join(SCRIPT_DIR, "results", "benchmark_results.csv")
    fieldnames = ["method", "learning_type", "autonomy_days", "energy_per_day_J",
                  "energy_J", "transmissions", "tx_per_mJ", "dropped_packets",
                  "useful_packets", "failures", "decision_us_avg",
                  "memory_B", "train_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\n[OK] Results saved to {csv_path}")

    return results


if __name__ == "__main__":
    main()
