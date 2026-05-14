"""
e3_comm_cost.py
---------------
Experiment 3: Communication cost — H-FL vs Flat FedAvg.

Analytical computation only — no model training.

Communication model per round:
    Flat FedAvg : every node sends its full model to the global server.
                  cost = N_nodes * params

    H-FL        : two-tier aggregation.
                  Tier 1 (intra-cluster): each node sends to its cluster head.
                            cost_tier1 = N_nodes * params
                  Tier 2 (inter-cluster): each cluster head sends to global.
                            cost_tier2 = N_clusters * params
                  total  = (N_nodes + N_clusters) * params

    Note: H-FL total transfers are slightly *more* than Flat at small N,
    but Tier 1 traffic stays within a local cluster (lower-bandwidth tier),
    and Tier 2 traffic (long-haul) is reduced from N to K transmissions.
    The plot annotates long-haul (Tier 2) cost separately.

Sweep: N_nodes from 10 to 100, clusters = ceil(N / 3).

Outputs:
    data/results/e3_comm_cost.csv
    data/plots/e3_comm_cost.png
"""

import os
import sys
import csv
import math

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
import config

# Model parameter count from known architecture:
# n_users=943, n_items=1349, emb_dim=64 -> embedding table = (943+1349)*64 = 146,688 params
# (GNN has no extra weight params in LightGCN)
MODEL_PARAMS = (943 + 1349) * 64  # 146,688


def run(save_plot: bool = True) -> list:
    """
    Returns list of dicts with comm cost data per N.
    """
    node_counts = list(range(10, 101, 10))
    rows = []

    for n in node_counts:
        k = math.ceil(n / 3)  # number of clusters

        flat_total    = n * MODEL_PARAMS          # all nodes -> global
        hfl_tier1     = n * MODEL_PARAMS          # nodes -> cluster heads
        hfl_tier2     = k * MODEL_PARAMS          # cluster heads -> global
        hfl_total     = hfl_tier1 + hfl_tier2

        # Long-haul saving: H-FL Tier2 vs Flat total
        longhual_saving_pct = round((1 - hfl_tier2 / flat_total) * 100, 2)

        rows.append({
            "n_nodes":              n,
            "n_clusters":           k,
            "model_params":         MODEL_PARAMS,
            "flat_params_total":    flat_total,
            "hfl_tier1_params":     hfl_tier1,
            "hfl_tier2_params":     hfl_tier2,
            "hfl_total_params":     hfl_total,
            "longhual_saving_pct":  longhual_saving_pct,
        })

    # ── Save CSV ─────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "e3_comm_cost.csv")
    fieldnames = [
        "n_nodes", "n_clusters", "model_params",
        "flat_params_total", "hfl_tier1_params", "hfl_tier2_params",
        "hfl_total_params", "longhual_saving_pct",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if save_plot:
        _plot(rows)

    return rows


def _plot(rows: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ns          = [r["n_nodes"]           for r in rows]
    flat_total  = [r["flat_params_total"]  / 1e6 for r in rows]  # millions
    hfl_total   = [r["hfl_total_params"]   / 1e6 for r in rows]
    hfl_tier2   = [r["hfl_tier2_params"]   / 1e6 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ns, flat_total, "s--", color="#E53935", linewidth=2.0,
            markersize=6, label="Flat FedAvg (total)")
    ax.plot(ns, hfl_total,  "o-",  color="#1565C0", linewidth=2.0,
            markersize=6, label="H-FL (total: Tier1 + Tier2)")
    ax.plot(ns, hfl_tier2,  "^:",  color="#42A5F5", linewidth=1.5,
            markersize=5, label="H-FL Tier2 only (long-haul)")

    # Mark current system config (N=9)
    current = next((r for r in rows if r["n_nodes"] == 10), rows[0])
    ax.axvline(x=9, color="gray", linestyle=":", linewidth=1,
               label="Current config (N=9)")

    ax.set_xlabel("Number of Edge Nodes (N)", fontsize=12)
    ax.set_ylabel("Parameters Transferred per Round (millions)", fontsize=11)
    ax.set_title("E3 — Communication Cost: H-FL vs Flat FedAvg", fontsize=13)
    ax.set_xticks(ns)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out = os.path.join(plots_dir, "e3_comm_cost.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[E3] Plot saved → {out}")


if __name__ == "__main__":
    results = run()
    print("[E3] Communication Cost (params transferred per round):")
    print(f"  {'N':>4}  {'K':>4}  {'Flat':>12}  {'H-FL Total':>12}  "
          f"{'H-FL Tier2':>12}  {'Saving%':>8}")
    for r in results:
        print(f"  {r['n_nodes']:>4}  {r['n_clusters']:>4}  "
              f"{r['flat_params_total']:>12,}  {r['hfl_total_params']:>12,}  "
              f"{r['hfl_tier2_params']:>12,}  {r['longhual_saving_pct']:>7.1f}%")
