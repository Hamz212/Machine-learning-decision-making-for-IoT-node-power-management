"""
Plot benchmark results as a comparative bar chart.
Reads results/benchmark_results.csv and produces a multi-panel figure.
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results/benchmark_results.csv"
OUT_PATH = "results/benchmark_comparison.png"


def load_results(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_results(CSV_PATH)
    methods = [r["method"] for r in rows]
    autonomy = [float(r["autonomy_days"]) for r in rows]
    tx       = [float(r["transmissions"]) for r in rows]
    fails    = [float(r["failures"]) for r in rows]
    drops    = [float(r["dropped_packets"]) for r in rows]
    dec_us   = [float(r["decision_us_avg"]) for r in rows]
    mem      = [float(r["memory_B"]) for r in rows]

    # Color coding by learning type
    type_colors = {
        "none": "#9E9E9E",
        "rule-based": "#1F77B4",
        "rule-based (hand-crafted)": "#1F77B4",
        "optimization": "#FF7F0E",
        "reinforcement": "#7030A0",
        "supervised (from RL oracle)": "#2CA02C",
        "hybrid (RL→supervised)": "#D62728",
    }
    colors = [type_colors.get(r["learning_type"], "#888") for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Benchmark des méthodes de décision — Power Manager IoT",
                 fontsize=14, fontweight="bold", y=0.995)

    # Panel 1: Successful transmissions
    ax = axes[0, 0]
    bars = ax.barh(methods, tx, color=colors)
    ax.set_xlabel("Transmissions réussies (sur 7 jours)")
    ax.set_title("Qualité de service")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, tx):
        ax.text(v, bar.get_y()+bar.get_height()/2,
                f" {int(v)}", va="center", fontsize=9)

    # Panel 2: Dropped packets (lower is better)
    ax = axes[0, 1]
    bars = ax.barh(methods, drops, color=colors)
    ax.set_xlabel("Packets perdus")
    ax.set_title("Perte de données (plus bas = mieux)")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, drops):
        ax.text(v, bar.get_y()+bar.get_height()/2,
                f" {int(v)}", va="center", fontsize=9)

    # Panel 3: Decision time
    ax = axes[1, 0]
    bars = ax.barh(methods, dec_us, color=colors)
    ax.set_xlabel("Temps moyen par décision (µs)")
    ax.set_title("Coût computationnel à l'inférence")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, dec_us):
        ax.text(v, bar.get_y()+bar.get_height()/2,
                f" {v:.1f}", va="center", fontsize=9)

    # Panel 4: Memory footprint
    ax = axes[1, 1]
    bars = ax.barh(methods, mem, color=colors)
    ax.set_xlabel("Empreinte mémoire (octets)")
    ax.set_title("Coût déploiement (FPGA / ASIC)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xscale("log")
    for bar, v in zip(bars, mem):
        ax.text(v, bar.get_y()+bar.get_height()/2,
                f" {int(v)} B", va="center", fontsize=9)

    # Legend
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=t)
        for t, c in type_colors.items()
        if t in [r["learning_type"] for r in rows]
    ]
    fig.legend(handles=legend_items, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=9,
               title="Type d'apprentissage")

    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"✓ Figure saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
