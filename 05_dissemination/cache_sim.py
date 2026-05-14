"""
cache_sim.py
------------
Content dissemination simulator for H-GNN-Consense.

Pipeline:
    1. DemandScorer   — scores all items for each edge node using the
                        global LightGCN model
    2. CacheSimulator — populates a top-K cache per node, then simulates
                        hit/miss against real test interactions
    3. run_cache_simulation() — ties both together, saves results

Inputs:
    data/models/global_lightgcn.pt
    data/models/node_{i}_meta.json      (for architecture params)
    data/processed/stats.json
    data/processed/node_user_map.json
    data/processed/test.csv

Outputs:
    data/results/cache_results.json     — per-node hit/miss counts
    data/results/cache_summary.json     — overall hit rate + config

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \\
    python 05_dissemination/cache_sim.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "03_gnn"))

import config
from lightgcn import LightGCN


def log(msg: str):
    print(f"[cache_sim] {msg}", flush=True)


# ----------------------------------------------------------------
# DemandScorer
# ----------------------------------------------------------------

class DemandScorer:
    """
    Scores all items for a given set of users using a LightGCN model.

    The demand score for item j on a node is the mean predicted
    relevance across all users assigned to that node:

        demand[j] = mean_u ( user_emb[u] · item_emb[j] )

    This reflects how popular item j is expected to be on this node.
    """

    def __init__(self, model: LightGCN):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        user_ids: List[int],
    ) -> np.ndarray:
        """
        Compute demand scores for all items given a list of user IDs.

        Args:
            user_ids : list of global user IDs assigned to this node

        Returns:
            np.ndarray of shape [n_items], float32 — mean score per item
        """
        if not user_ids:
            raise ValueError("DemandScorer.score(): user_ids list is empty.")

        users_emb, items_emb = self.model()       # [n_users, D], [n_items, D]

        # Select only this node's users
        user_t  = torch.tensor(user_ids, dtype=torch.long)
        node_u  = users_emb[user_t]               # [|node_users|, D]

        # Mean user embedding for this node
        mean_u  = node_u.mean(dim=0, keepdim=True)  # [1, D]

        # Score all items: [1, D] x [D, n_items] -> [n_items]
        scores  = (mean_u @ items_emb.T).squeeze(0)  # [n_items]

        return scores.cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------
# CacheSimulator
# ----------------------------------------------------------------

class CacheSimulator:
    """
    Simulates a proactive content cache at each edge node.

    The cache holds the top-K items by predicted demand score.
    Hit/miss is evaluated by checking whether each test interaction's
    item is in the node's cache.
    """

    def __init__(self, cache_size_k: int = 50):
        self.cache_size_k = cache_size_k
        self._caches: Dict[int, np.ndarray] = {}   # node_id -> top-K item IDs

    def populate(
        self,
        node_id: int,
        demand_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Fills the cache for a node with top-K items by demand score.

        Args:
            node_id       : edge node ID
            demand_scores : np.ndarray [n_items] — score per item

        Returns:
            np.ndarray of top-K item IDs (sorted by score, descending)
        """
        k = min(self.cache_size_k, len(demand_scores))
        # argpartition is O(n) — faster than full argsort for large n_items
        top_k_idx = np.argpartition(demand_scores, -k)[-k:]
        # Sort within top-K by score descending
        top_k_sorted = top_k_idx[np.argsort(demand_scores[top_k_idx])[::-1]]
        self._caches[node_id] = top_k_sorted
        return top_k_sorted

    def simulate_requests(
        self,
        node_id: int,
        test_interactions: pd.DataFrame,
    ) -> Dict:
        """
        Simulates content requests from test interactions.

        For each (user, item) pair in test_interactions:
            HIT  if item_id is in this node's cache
            MISS otherwise

        Args:
            node_id           : edge node ID
            test_interactions : DataFrame with column 'item_id'

        Returns:
            dict with keys: hits, misses, total, hit_rate, cached_items
        """
        if node_id not in self._caches:
            raise ValueError(f"Node {node_id} cache not populated. Call populate() first.")

        cached = set(self._caches[node_id].tolist())
        requested_items = test_interactions["item_id"].values

        hits   = int(np.isin(requested_items, list(cached)).sum())
        total  = len(requested_items)
        misses = total - hits

        return {
            "node_id":      node_id,
            "hits":         hits,
            "misses":       misses,
            "total":        total,
            "hit_rate":     round(hits / total, 6) if total > 0 else 0.0,
            "cache_size":   len(cached),
            "cached_items": self._caches[node_id].tolist(),
        }


# ----------------------------------------------------------------
# Load global model
# ----------------------------------------------------------------

def load_global_model(n_users, n_items, device):
    """
    Loads the global LightGCN model saved by Module 4.
    Falls back to node_0_lightgcn.pt if global not found.
    """
    global_path = os.path.join(config.MODELS_DIR, "global_lightgcn.pt")
    fallback    = os.path.join(config.MODELS_DIR, "node_0_lightgcn.pt")
    meta_path   = os.path.join(config.MODELS_DIR, "node_0_meta.json")

    if os.path.exists(global_path):
        model_path = global_path
        log(f"Loading global model: {global_path}")
    elif os.path.exists(fallback):
        model_path = fallback
        log(f"WARNING: global_lightgcn.pt not found. Using {fallback} as fallback.")
        log("         Run Module 4 (fl_runner.py) for federated results.")
    else:
        raise FileNotFoundError(
            "No model found. Run Module 3 and optionally Module 4 first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    dummy_src = np.array([0], dtype=np.int64)
    dummy_dst = np.array([0], dtype=np.int64)
    model = LightGCN(
        n_users  = n_users,
        n_items  = n_items,
        emb_dim  = meta["emb_dim"],
        n_layers = meta["n_layers"],
        edge_src = dummy_src,
        edge_dst = dummy_dst,
    ).to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    edge_keys = ["edge_src", "edge_dst", "edge_weights"]
    for k in edge_keys:
        if k in state:
            setattr(model, k, state[k].to(device))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ----------------------------------------------------------------
# Main simulation runner
# ----------------------------------------------------------------

def run_cache_simulation():
    t_start = time.time()

    log("=" * 60)
    log("Module 5 — Cache Dissemination Simulation")
    log("=" * 60)

    device = torch.device("cpu")

    # ---- Load inputs ----
    stats_path    = os.path.join(config.PROC_DATA_DIR, "stats.json")
    node_map_path = os.path.join(config.PROC_DATA_DIR, "node_user_map.json")
    test_path     = os.path.join(config.PROC_DATA_DIR, "test.csv")

    for p in [stats_path, node_map_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}. Run Module 1 first.")

    with open(stats_path) as f:
        stats = json.load(f)
    with open(node_map_path) as f:
        node_user_map = json.load(f)

    test_df  = pd.read_csv(test_path)
    n_users  = stats["n_users"]
    n_items  = stats["n_items"]

    log(f"Users: {n_users}  Items: {n_items}  Cache size K: {config.CACHE_SIZE_K}")
    log(f"Test interactions: {len(test_df):,}")

    # ---- Load global model ----
    model   = load_global_model(n_users, n_items, device)
    scorer  = DemandScorer(model)
    cache   = CacheSimulator(cache_size_k=config.CACHE_SIZE_K)

    # ---- Simulate per node ----
    node_results = []
    node_ids     = sorted([int(k) for k in node_user_map.keys()])

    for nid in node_ids:
        user_ids = [int(u) for u in node_user_map.get(str(nid), [])]
        if not user_ids:
            log(f"  Node {nid}: no users — skipping")
            continue

        log(f"  Node {nid}: {len(user_ids)} users — scoring items...")

        # Step 1: Score all items for this node's users
        demand_scores = scorer.score(user_ids)

        # Step 2: Populate cache with top-K
        top_k = cache.populate(nid, demand_scores)
        log(f"    Cached items: {top_k[:5].tolist()} ... (top 5 of {len(top_k)})")

        # Step 3: Simulate requests from test interactions
        node_test = test_df[test_df["user_id"].isin(user_ids)]
        result    = cache.simulate_requests(nid, node_test)

        log(f"    Requests: {result['total']:,}  "
            f"Hits: {result['hits']:,}  "
            f"Misses: {result['misses']:,}  "
            f"Hit Rate: {result['hit_rate']:.4f}")

        node_results.append(result)

    # ---- Aggregate overall hit rate ----
    total_hits   = sum(r["hits"]   for r in node_results)
    total_reqs   = sum(r["total"]  for r in node_results)
    overall_rate = round(total_hits / total_reqs, 6) if total_reqs > 0 else 0.0

    log(f"\nOverall — Requests: {total_reqs:,}  "
        f"Hits: {total_hits:,}  Hit Rate: {overall_rate:.4f}")

    # ---- Save results ----
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(config.RESULTS_DIR, "cache_results.json")
    # Don't save full cached_items list in main results (too large) —
    # keep only summary per node
    slim_results = [
        {k: v for k, v in r.items() if k != "cached_items"}
        for r in node_results
    ]
    with open(results_path, "w") as f:
        json.dump(slim_results, f, indent=2)
    log(f"Node results   → {results_path}")

    summary = {
        "cache_size_k":    config.CACHE_SIZE_K,
        "n_nodes":         len(node_results),
        "total_requests":  total_reqs,
        "total_hits":      total_hits,
        "overall_hit_rate": overall_rate,
        "per_node_hit_rates": {
            str(r["node_id"]): r["hit_rate"] for r in node_results
        },
        "model_used": "global_lightgcn.pt" if os.path.exists(
            os.path.join(config.MODELS_DIR, "global_lightgcn.pt")
        ) else "node_0_lightgcn.pt (fallback)",
        "completed_at": datetime.utcnow().isoformat(),
        "total_time_sec": round(time.time() - t_start, 1),
    }

    summary_path = os.path.join(config.RESULTS_DIR, "cache_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary        → {summary_path}")

    log("\n" + "=" * 60)
    log("Module 5 complete")
    log(f"  Overall hit rate : {overall_rate:.4f} ({overall_rate*100:.1f}%)")
    log(f"  Cache size K     : {config.CACHE_SIZE_K}")
    log(f"  Total time       : {time.time() - t_start:.1f}s")
    log("=" * 60)

    return node_results, summary


if __name__ == "__main__":
    run_cache_simulation()
