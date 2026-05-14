"""
e4_scalability.py
-----------------
Experiment 4: Scalability — cache hit rate as N (edge nodes) scales 10→90.

Uses the already-trained global LightGCN model (no retraining).
Synthetically repartitions the 943 users across N nodes for each N.
Runs the cache simulator (K=50) for each partition.

Sweep: N in {10, 20, 30, 40, 50, 60, 70, 80, 90}
  (100 excluded: 943 users / 100 = ~9 users/node, too sparse for meaningful eval)

Outputs:
    data/results/e4_scalability.csv
    data/plots/e4_scalability.png
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
    """Load global LightGCN using same pattern as cache_sim.py."""
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
    return model, meta["n_users"], meta["n_items"]


def _simulate_for_n(model, n_nodes: int, test_df, all_user_ids: list,
                   n_items: int, cache_k: int = 50) -> dict:
    """
    Partition users across n_nodes, run DemandScorer + CacheSimulator,
    return overall hit rate.
    """
    import numpy as np
    from cache_sim import DemandScorer, CacheSimulator

    scorer    = DemandScorer(model)
    simulator = CacheSimulator(cache_size_k=cache_k)

    # Round-robin partition: user i -> node (i % n_nodes)
    node_users = {nid: [] for nid in range(n_nodes)}
    for i, uid in enumerate(all_user_ids):
        node_users[i % n_nodes].append(uid)

    total_hits = 0
    total_reqs = 0

    for nid in range(n_nodes):
        users = node_users[nid]
        if not users:
            continue

        # Score items for this node's users
        scores = scorer.score(users)
        simulator.populate(nid, scores)

        # Simulate requests: test interactions for this node's users
        node_test = test_df[test_df["user_id"].isin(users)]
        if node_test.empty:
            continue

        result = simulator.simulate_requests(nid, node_test)
        total_hits += result["hits"]
        total_reqs += result["total"]

    hit_rate = round(total_hits / total_reqs, 6) if total_reqs > 0 else 0.0
    return {"total_hits": total_hits, "total_reqs": total_reqs, "hit_rate": hit_rate}


def run(save_plot: bool = True) -> list:
    """
    Returns list of dicts: [{n_nodes, hit_rate, total_requests, total_hits, time_sec}, ...]
    """
    import pandas as pd

    test_path = os.path.join(config.PROC_DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing {test_path} — run Module 1 first.")

    test_df = pd.read_csv(test_path)
    all_user_ids = sorted(test_df["user_id"].unique().tolist())

    model, n_users, n_items = _load_global_model()

    node_counts = list(range(10, 91, 10))  # 10,20,...,90
    rows = []

    for n in node_counts:
        t0  = time.perf_counter()
        res = _simulate_for_n(model, n, test_df, all_user_ids, n_items)
        elapsed = round(time.perf_counter() - t0, 3)

        rows.append({
            "n_nodes":        n,
            "hit_rate":       res["hit_rate"],
            "total_requests": res["total_reqs"],
            "total_hits":     res["total_hits"],
            "time_sec":       elapsed,
        })
        print(f"[E4]   N={n:>3}  hit_rate={res['hit_rate']:.4f}  "
              f"({res['total_hits']}/{res['total_reqs']})  {elapsed:.2f}s")

    # ── Save CSV ─────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "e4_scalability.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "n_nodes", "hit_rate", "total_requests", "total_hits", "time_sec"
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

    plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ns        = [r["n_nodes"]  for r in rows]
    hit_rates = [r["hit_rate"] * 100 for r in rows]

    # Compute mean +/- band to visualise stability
    import numpy as np
    mean_hr = float(np.mean(hit_rates))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, hit_rates, "o-", color="#1565C0", linewidth=2.0, markersize=7)
    ax.axhline(mean_hr, color="gray", linestyle="--", linewidth=1,
               label=f"Mean hit rate ({mean_hr:.1f}%)")

    # Shade between min and max across all N
    ax.fill_between(ns,
                    [min(hit_rates)] * len(ns),
                    [max(hit_rates)] * len(ns),
                    alpha=0.08, color="#1565C0")

    # Annotate N=9 (actual system)
    ax.axvline(x=9, color="#E53935", linestyle=":", linewidth=1.2,
               label="Actual deployment (N=9)")

    for x, y in zip(ns, hit_rates):
        ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Number of Edge Nodes (N)", fontsize=12)
    ax.set_ylabel("Cache Hit Rate (%)", fontsize=12)
    ax.set_title("E4 — Scalability: Hit Rate vs Number of Edge Nodes", fontsize=13)
    ax.set_xticks(ns)
    ax.set_ylim(0, max(hit_rates) * 1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out = os.path.join(plots_dir, "e4_scalability.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[E4] Plot saved → {out}")


if __name__ == "__main__":
    results = run()
    print("\n[E4] Scalability Results:")
    for r in results:
        print(f"  N={r['n_nodes']:>3}  hit_rate={r['hit_rate']*100:.2f}%  time={r['time_sec']:.2f}s")
