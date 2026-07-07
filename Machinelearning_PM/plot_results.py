"""
Plot benchmark v2 results.
Focus : autonomy, energy consumption, efficiency.
"""

import os
import csv
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_results.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_comparison.png")


def load_results(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERREUR] Fichier CSV introuvable : {CSV_PATH}")
        print("Lance d'abord : python3 run_benchmark.py")
        return

    rows = load_results(CSV_PATH)
    # Sort ascending for horizontal bars (best at top)
    rows_by_auto = sorted(rows, key=lambda r: float(r["autonomy_days"]))

    methods_auto = [r["method"] for r in rows_by_auto]
    autonomy    = [float(r["autonomy_days"]) for r in rows_by_auto]
    e_per_day   = [float(r["energy_per_day_J"]) for r in rows_by_auto]
    tx_per_mJ   = [float(r["tx_per_mJ"]) for r in rows_by_auto]
    tx          = [float(r["transmissions"]) for r in rows_by_auto]
    mem         = [float(r["memory_B"]) for r in rows_by_auto]

    type_colors = {
        "none": "#9E9E9E",
        "rule-based": "#1F77B4",
        "rule-based (hand-crafted)": "#1F77B4",
        "optimization": "#FF7F0E",
        "reinforcement": "#7030A0",
        "supervised (from RL oracle)": "#2CA02C",
        "hybrid (RL->supervised)": "#D62728",
    }
    colors = [type_colors.get(r["learning_type"], "#888") for r in rows_by_auto]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Benchmark v2 -- OBJECTIF AUTONOMIE MAXIMALE (60 jours TVCC)",
                 fontsize=14, fontweight="bold", y=0.995)

    # Panel 1: AUTONOMY (le KPI principal)
    ax = axes[0, 0]
    bars = ax.barh(methods_auto, autonomy, color=colors)
    ax.set_xlabel("Jours de survie avant power failure")
    ax.set_title("[KPI PRINCIPAL] Autonomie", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(x=60, color="black", linestyle="--", alpha=0.5,
               label="Objectif TVCC : 60 jours")
    ax.legend(loc="lower right", fontsize=8)
    for bar, v in zip(bars, autonomy):
        ax.text(v, bar.get_y()+bar.get_height()/2,
                f" {v:.1f}d", va="center", fontsize=9, fontweight="bold")

    # Panel 2: Energy consumed per day (efficiency)
    ax = axes[0, 1]
    bars = ax.barh(methods_auto, e_per_day, color=colors)
    ax.set_xlabel("Energie consommee par jour (J/jour)")
    ax.set_title("Cout energetique (plus bas = mieux)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, e_per_day):
        ax.text(v, bar.get_y()+bar.get_height()/2,
                f" {v:.2f}", va="center", fontsize=9)

    # Panel 3: TX efficiency (transmissions per mJ)
    ax = axes[1, 0]
    bars = ax.barh(methods_auto, tx_per_mJ, color=colors)
    ax.set_xlabel("Transmissions par J consomme")
    ax.set_title("Efficacite energetique des TX (plus haut = mieux)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, tx_per_mJ):
        ax.text(v, bar.get_y()+bar.get_height()/2,
                f" {v:.2f}", va="center", fontsize=9)

    # Panel 4: Memory footprint (deployment)
    ax = axes[1, 1]
    bars = ax.barh(methods_auto, mem, color=colors)
    ax.set_xlabel("Empreinte memoire (octets)")
    ax.set_title("Cout deploiement (FPGA / ASIC)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xscale("log")
    for bar, v in zip(bars, mem):
        ax.text(v, bar.get_y()+bar.get_height()/2,
                f" {int(v)} B", va="center", fontsize=9)

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
    print(f"[OK] Figure saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
