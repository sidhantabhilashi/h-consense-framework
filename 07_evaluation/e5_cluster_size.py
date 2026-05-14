"""
e5_cluster_size.py
------------------
Experiment 5: Cluster size effect — K = 2, 3, 4, 5.

Uses the already-trained global LightGCN model (no retraining).
For each K, re-clusters the 9 edge nodes into K groups using the
real node_user_map, then runs the cache simulator.

Cluster assignment: simple round-robin grouping of the 9 nodes
(node i -> cluster i % K). This is deterministic and reproducible.
In the full thesis, you can also run with spectral clustering.

N=9 nodes fixed, K in {2, 3, 4, 5}.

Outputs:
    data/results/e5_cluster_size.csv
    data/plots/e5_cluster_size.png
"""

import os
import sys
import csv
import time
import json

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "03_gnn"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "05_dissemination"))
import config


def _load_global_model():
    """Load global LightGCN (same pattern as cache_sim.py / e4)."""
    import numpy as np
    import torch
    from lightgcn import LightGCN

    meta_path  = os.path.join(config.MODELS_DIR, "node_0_meta.json")
    model_path = os.path.join(config.MODELS_DIR, "global_lightgcn.pt")

    with open(meta_path) as f:
        meta = json.load(f)

    dummy = np.array([0], dtype=np.int64)
    model = LightGCN(
        n_users  = meta["n_users"],
        n_items  = meta["n_items"],
        emb_dim  = meta["emb_dim"],
        n_layers = meta["n_layers"],
        edge_src = dummy,
        edge_dst = dummy,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    # Strip graph buffers before load_state_dict to avoid PyTorch 2.x size-mismatch
    # errors on buffers registered with dummy shape during __init__.
    _es = state.pop("edge_src",     None)
    _ed = state.pop("edge_dst",     None)
    _ew = state.pop("edge_weights", None)
    model.load_state_dict(state, strict=False)
    if _es is not None:
        model.register_buffer("edge_src", _es)
        model.register_buffer("edge_dst", _ed)
    if _ew is not None:
        model.register_buffer("edge_weights", _ew)
    model.eval()
    return model


def _simulate_for_k(model, k_clusters: int, node_user_map: dict,
                   test_df, cache_k: int = 50) -> dict:
    """
    Run cache simulation with K cluster grouping.
    Returns hit rate and comm cost per round.
    """
    import math
    import numpy as np
    from cache_sim import DemandScorer, CacheSimulator

    n_nodes   = len(node_user_map)
    scorer    = DemandScorer(model)
    simulator = CacheSimulator(cache_size_k=cache_k)

    total_hits = 0
    total_reqs = 0

    # Build cluster->node mapping (round-robin assignment)
    cluster_nodes = {c: [] for c in range(k_clusters)}
    for nid in range(n_nodes):
        cluster_nodes[nid % k_clusters].append(nid)

    for nid in range(n_nodes):
        users = node_user_map.get(nid, [])
        if not users:
            continue

        scores = scorer.score(users)
        simulator.populate(nid, scores)

        node_test = test_df[test_df["user_id"].isin(users)]
        if node_test.empty:
            continue

        result = simulator.simulate_requests(nid, node_test)
        total_hits += result["hits"]
        total_reqs += result["total"]

    hit_rate = round(total_hits / total_reqs, 6) if total_reqs > 0 else 0.0

    # Communication cost per FL round for this K
    model_params     = (943 + 1349) * 64          # 146,688
    comm_tier1       = n_nodes    * model_params   # nodes -> cluster heads
    comm_tier2       = k_clusters * model_params   # cluster heads -> global
    comm_total       = comm_tier1 + comm_tier2

    return {
        "total_hits":   total_hits,
        "total_reqs":   total_reqs,
        "hit_rate":     hit_rate,
        "comm_total":   comm_total,
        "comm_tier2":   comm_tier2,
    }


def run(save_plot: bool = True) -> list:
    """
    Returns list of dicts: [{k_clusters, hit_rate, comm_total, comm_tier2, time_sec}, ...]
    """
    import pandas as pd

    test_path  = os.path.join(config.PROC_DATA_DIR, "test.csv")
    nmap_path  = os.path.join(config.PROC_DATA_DIR, "node_user_map.json")

    for p in [test_path, nmap_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p} — run Module 1 first.")

    test_df = pd.read_csv(test_path)
    with open(nmap_path) as f:
        node_user_map = {int(k): v for k, v in json.load(f).items()}

    model = _load_global_model()

    k_values = [2, 3, 4, 5]
    rows = []

    for k in k_values:
        t0  = time.perf_counter()
        res = _simulate_for_k(model, k, node_user_map, test_df)
        elapsed = round(time.perf_counter() - t0, 3)

        rows.append({
            "k_clusters":   k,
            "hit_rate":     res["hit_rate"],
            "comm_total":   res["comm_total"],
            "comm_tier2":   res["comm_tier2"],
            "time_sec":     elapsed,
        })
        print(f"[E5]   K={k}  hit_rate={res['hit_rate']:.4f}  "
              f"comm_total={res['comm_total']:,}  {elapsed:.2f}s")

    # ── Save CSV ─────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "e5_cluster_size.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "k_clusters", "hit_rate", "comm_total", "comm_tier2", "time_sec"
        ])
        w.writeheader()
        w.writerows(rows)

    if save_plot:
        _plot(rows)

    return rows


def _plot(rows: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ks        = [r["k_clusters"] for r in rows]
    hit_rates = [r["hit_rate"] * 100 for r in rows]
    colors    = ["#90CAF9" if k != 3 else "#E53935" for k in ks]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([str(k) for k in ks], hit_rates, color=colors, width=0.55, zorder=3)

    for bar, val, k in zip(bars, hit_rates, ks):
        label = f"{val:.2f}%" + (" ← current" if k == 3 else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                label, ha="center", va="bottom", fontsize=9,
                fontweight="bold" if k == 3 else "normal")

    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Cache Hit Rate (%)", fontsize=12)
    ax.set_title("E5 — Cluster Size Effect on Cache Hit Rate", fontsize=13)
    ax.set_ylim(0, max(hit_rates) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend patch for current config
    import matplotlib.patches as mpatches
    current_patch = mpatches.Patch(color="#E53935", label="Current config (K=3)")
    ax.legend(handles=[current_patch], fontsize=9)
    plt.tight_layout()

    out = os.path.join(plots_dir, "e5_cluster_size.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[E5] Plot saved → {out}")


if __name__ == "__main__":
    results = run()
    print("\n[E5] Cluster Size Results:")
    for r in results:
        print(f"  K={r['k_clusters']}  hit_rate={r['hit_rate']*100:.2f}%  "
              f"comm={r['comm_total']:,}  {r['time_sec']:.2f}s")
