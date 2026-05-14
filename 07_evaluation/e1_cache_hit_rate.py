"""
e1_cache_hit_rate.py
--------------------
Experiment 1: Cache hit rate comparison.

Reads already-computed hit rates from M5 and M6 result JSONs.
No re-training or re-simulation.

Outputs:
    data/results/e1_cache_hit_rate.csv
    data/plots/e1_cache_hit_rate.png
"""

import os
import sys
import json
import csv

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
import config


def run(save_plot: bool = True) -> list:
    """
    Returns list of dicts: [{method, hit_rate, vs_random}, ...]
    """
    # ── Load source JSONs ────────────────────────────────────────────
    baseline_path = os.path.join(config.RESULTS_DIR, "baseline_summary.json")
    cache_path    = os.path.join(config.RESULTS_DIR, "cache_summary.json")

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Missing {baseline_path} — run Module 6 first.")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing {cache_path} — run Module 5 first.")

    with open(baseline_path) as f:
        bl = json.load(f)
    with open(cache_path) as f:
        cs = json.load(f)

    random_hr = bl["random_baseline"]

    rows = [
        {"method": "LRU (recency)",        "hit_rate": bl["lru_hit_rate"]},
        {"method": "LFU (popularity)",      "hit_rate": bl["lfu_hit_rate"]},
        {"method": "Flat FedAvg",           "hit_rate": bl["flat_fedavg_hit_rate"]},
        {"method": "H-GNN-Consense (ours)", "hit_rate": cs["overall_hit_rate"]},
    ]
    for r in rows:
        r["vs_random"] = round(r["hit_rate"] / random_hr, 3) if random_hr > 0 else 0.0

    # ── Save CSV ─────────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "e1_cache_hit_rate.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "hit_rate", "vs_random"])
        w.writeheader()
        w.writerows(rows)

    # ── Plot ─────────────────────────────────────────────────────────
    if save_plot:
        _plot(rows, random_hr)

    return rows


def _plot(rows: list, random_hr: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    methods   = [r["method"]   for r in rows]
    hit_rates = [r["hit_rate"] * 100 for r in rows]  # convert to %

    colors = ["#90CAF9", "#42A5F5", "#1565C0", "#E53935"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, hit_rates, color=colors, width=0.55, zorder=3)

    # Random baseline reference line
    ax.axhline(random_hr * 100, color="gray", linestyle="--", linewidth=1.2,
               label=f"Random baseline ({random_hr*100:.1f}%)", zorder=2)

    # Value labels on bars
    for bar, val in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Cache Hit Rate (%)", fontsize=12)
    ax.set_title("E1 — Cache Hit Rate: H-GNN-Consense vs Baselines", fontsize=13)
    ax.set_ylim(0, max(hit_rates) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.xticks(fontsize=9)
    plt.tight_layout()

    out = os.path.join(plots_dir, "e1_cache_hit_rate.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[E1] Plot saved → {out}")


if __name__ == "__main__":
    results = run()
    print("[E1] Cache Hit Rate Results:")
    for r in results:
        print(f"  {r['method']:<28} {r['hit_rate']*100:.2f}%  ({r['vs_random']:.2f}x random)")
