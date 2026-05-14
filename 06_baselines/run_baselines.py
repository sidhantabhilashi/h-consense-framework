"""
run_baselines.py
----------------
Orchestrates all three Module 6 baselines and saves comparison results.

Baselines:
    1. LFU  — top-K most frequent items from training data (popularity cache)
    2. LRU  — warm cache from training interactions, evict least-recently-used
    3. Flat FedAvg — standard single-tier FedAvg + demand-score cache

All baselines use:
    - Same CACHE_SIZE_K = 50 as Module 5
    - Same per-node train/test splits
    - Same result dict schema as Module 5 CacheSimulator

Outputs:
    data/results/baseline_results.json   — per-baseline, per-node results
    data/results/baseline_summary.json   — overall hit rates + comparison

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \\
    python 06_baselines/run_baselines.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",  "1")

import sys
import json
import time
import copy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "03_gnn"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "04_federated"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "05_dissemination"))
sys.path.insert(0, THIS_DIR)

import config
from lightgcn    import LightGCN
from cache_sim   import DemandScorer, CacheSimulator
from lru         import LRUCache
from lfu         import LFUCache
from flat_fedavg import FlatFedAvg

DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg: str):
    print(f"[baselines] {msg}", flush=True)


# ----------------------------------------------------------------
# Shared data loading (mirrors fl_runner.py)
# ----------------------------------------------------------------

def load_inputs(device: torch.device):
    """Load all shared inputs needed by the baselines."""
    stats           = json.load(open(os.path.join(DATA_DIR, "processed", "stats.json")))
    node_user_map   = json.load(open(os.path.join(DATA_DIR, "processed", "node_user_map.json")))
    n_users         = stats["n_users"]
    n_items         = stats["n_items"]

    train_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "processed", "test.csv"))

    # Per-node splits
    node_train_dfs: dict = {}
    node_test_dfs:  dict = {}
    for nid_str, user_ids in node_user_map.items():
        nid = int(nid_str)
        node_train_dfs[nid] = train_df[train_df["user_id"].isin(user_ids)].reset_index(drop=True)
        node_test_dfs[nid]  = test_df[ test_df["user_id"].isin(user_ids)].reset_index(drop=True)

    # Load per-node LightGCN models (same as fl_runner.py)
    node_models: dict = {}
    for nid in range(n_users if False else config.NUM_EDGE_NODES):
        meta_path  = os.path.join(MODELS_DIR, f"node_{nid}_meta.json")
        ckpt_path  = os.path.join(MODELS_DIR, f"node_{nid}_lightgcn.pt")
        meta       = json.load(open(meta_path))
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state      = checkpoint.get("model_state_dict", checkpoint)

        model = LightGCN(
            n_users  = meta["n_users"],
            n_items  = meta["n_items"],
            emb_dim  = meta["emb_dim"],
            n_layers = meta["n_layers"],
            edge_src = np.array(state["edge_src"].cpu()),
            edge_dst = np.array(state["edge_dst"].cpu()),
        ).to(device)

        for k in ("edge_src", "edge_dst", "edge_weights"):
            if k in state:
                setattr(model, k, state[k].to(device))
        model.load_state_dict(state, strict=False)
        model.eval()
        node_models[nid] = model

    return n_users, n_items, node_user_map, node_train_dfs, node_test_dfs, node_models


def _aggregate_results(per_node: list) -> float:
    """Compute overall weighted hit rate from per-node result dicts."""
    total_hits = sum(r["hits"]  for r in per_node)
    total_req  = sum(r["total"] for r in per_node)
    return total_hits / total_req if total_req > 0 else 0.0


# ----------------------------------------------------------------
# Baseline 1: LFU
# ----------------------------------------------------------------

def run_lfu_baseline(node_train_dfs, node_test_dfs) -> list:
    log("")
    log("--- Baseline 1: LFU (popularity-based) ---")
    results = []
    for nid in sorted(node_train_dfs.keys()):
        cache = LFUCache(cache_size_k=config.CACHE_SIZE_K)
        cache.warm(node_train_dfs[nid])
        result = cache.simulate_requests(nid, node_test_dfs[nid])
        log(f"  Node {nid}: Requests={result['total']:,}  "
            f"Hits={result['hits']:,}  Hit Rate={result['hit_rate']:.4f}")
        results.append(result)
    overall = _aggregate_results(results)
    log(f"  LFU Overall Hit Rate: {overall:.4f} ({overall*100:.1f}%)")
    return results


# ----------------------------------------------------------------
# Baseline 2: LRU
# ----------------------------------------------------------------

def run_lru_baseline(node_train_dfs, node_test_dfs) -> list:
    log("")
    log("--- Baseline 2: LRU (recency-based) ---")
    results = []
    for nid in sorted(node_train_dfs.keys()):
        cache = LRUCache(cache_size_k=config.CACHE_SIZE_K)
        cache.warm(node_train_dfs[nid])
        result = cache.simulate_requests(nid, node_test_dfs[nid])
        log(f"  Node {nid}: Requests={result['total']:,}  "
            f"Hits={result['hits']:,}  Hit Rate={result['hit_rate']:.4f}")
        results.append(result)
    overall = _aggregate_results(results)
    log(f"  LRU Overall Hit Rate: {overall:.4f} ({overall*100:.1f}%)")
    return results


# ----------------------------------------------------------------
# Baseline 3: Flat FedAvg
# ----------------------------------------------------------------

def run_flat_fedavg_baseline(
    node_models, node_train_dfs, node_test_dfs,
    n_items, device
) -> list:
    log("")
    log("--- Baseline 3: Flat FedAvg (no hierarchy) ---")

    # Deep-copy models so we don't corrupt the originals
    flat_models = {nid: copy.deepcopy(m) for nid, m in node_models.items()}

    fl = FlatFedAvg(
        n_rounds     = config.FL_ROUNDS,
        local_epochs = config.LOCAL_EPOCHS,
        device       = str(device),
    )
    global_model = fl.run(
        node_models    = flat_models,
        node_train_dfs = node_train_dfs,
        n_items        = n_items,
        log_fn         = log,
    )

    log("  Scoring demand and simulating cache...")
    scorer = DemandScorer(global_model)
    cache  = CacheSimulator(cache_size_k=config.CACHE_SIZE_K)

    node_user_map_local = {
        nid: node_train_dfs[nid]["user_id"].unique().tolist()
        for nid in node_train_dfs
    }

    results = []
    for nid in sorted(node_test_dfs.keys()):
        user_ids = node_user_map_local[nid]
        if not user_ids:
            continue
        scores = scorer.score(user_ids)
        cache.populate(nid, scores)
        result = cache.simulate_requests(nid, node_test_dfs[nid])
        log(f"  Node {nid}: Requests={result['total']:,}  "
            f"Hits={result['hits']:,}  Hit Rate={result['hit_rate']:.4f}")
        results.append(result)

    overall = _aggregate_results(results)
    log(f"  Flat FedAvg Overall Hit Rate: {overall:.4f} ({overall*100:.1f}%)")
    return results


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    t0     = time.time()
    device = torch.device("cpu")

    log("=" * 60)
    log("Module 6 — Baselines")
    log("=" * 60)

    log("Loading inputs...")
    n_users, n_items, node_user_map, node_train_dfs, node_test_dfs, node_models = \
        load_inputs(device)
    log(f"  Users={n_users}  Items={n_items}  "
        f"Nodes={len(node_train_dfs)}  Cache K={config.CACHE_SIZE_K}")

    # Run all three baselines
    lfu_results      = run_lfu_baseline(node_train_dfs, node_test_dfs)
    lru_results      = run_lru_baseline(node_train_dfs, node_test_dfs)
    flat_results     = run_flat_fedavg_baseline(
        node_models, node_train_dfs, node_test_dfs, n_items, device
    )

    # Load H-GNN-Consense result for comparison
    hgnn_hit_rate = None
    summary_path  = os.path.join(RESULTS_DIR, "cache_summary.json")
    if os.path.exists(summary_path):
        hgnn_hit_rate = json.load(open(summary_path)).get("overall_hit_rate")

    # Build comparison
    lfu_overall  = _aggregate_results(lfu_results)
    lru_overall  = _aggregate_results(lru_results)
    flat_overall = _aggregate_results(flat_results)

    log("")
    log("=" * 60)
    log("COMPARISON TABLE")
    log("=" * 60)
    log(f"  {'Method':<25} {'Hit Rate':>10}  {'vs Random':>12}")
    log(f"  {'-'*25}  {'-'*10}  {'-'*12}")
    random_baseline = config.CACHE_SIZE_K / n_items
    for name, rate in [
        ("LFU (popularity)",     lfu_overall),
        ("LRU (recency)",        lru_overall),
        ("Flat FedAvg",          flat_overall),
        ("H-GNN-Consense (M5)",  hgnn_hit_rate or 0.0),
    ]:
        lift = rate / random_baseline if random_baseline > 0 else 0
        log(f"  {name:<25} {rate*100:>9.2f}%  {lift:>10.2f}x")
    log("=" * 60)

    # Save outputs
    all_results = {
        "lfu":       [dict(r, baseline="lfu")  for r in lfu_results],
        "lru":       [dict(r, baseline="lru")  for r in lru_results],
        "flat_fedavg": [dict(r, baseline="flat_fedavg") for r in flat_results],
    }

    results_out = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(results_out, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"  Per-node results → {results_out}")

    summary = {
        "cache_size_k":          config.CACHE_SIZE_K,
        "random_baseline":       round(random_baseline, 6),
        "lfu_hit_rate":          round(lfu_overall, 6),
        "lru_hit_rate":          round(lru_overall, 6),
        "flat_fedavg_hit_rate":  round(flat_overall, 6),
        "hgnn_consense_hit_rate": round(hgnn_hit_rate, 6) if hgnn_hit_rate else None,
        "total_time_sec":        round(time.time() - t0, 1),
        "completed_at":          datetime.now(timezone.utc).isoformat(),
    }
    summary_out = os.path.join(RESULTS_DIR, "baseline_summary.json")
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Summary          → {summary_out}")

    log("")
    log("Module 6 complete")
    log(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
