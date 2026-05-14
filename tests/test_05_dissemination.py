"""
test_05_dissemination.py
------------------------
Unit tests for Module 5 (05_dissemination/cache_sim.py).

Runs WITHOUT real data, saved models, or disk I/O.
All tests use tiny synthetic LightGCN models and in-memory DataFrames.
Fast: should complete in < 10 seconds on CPU.

Usage (run from the code/ directory):
    pytest tests/test_05_dissemination.py -v

What is tested:
    DemandScorer:
        - output shape is [n_items]
        - output dtype is float32
        - no NaN or Inf in scores
        - different user groups produce different scores
        - scores vary across items (not all identical)
        - empty user list raises an error

    CacheSimulator:
        - populate() returns exactly K items (or n_items if K > n_items)
        - cached items have higher scores than non-cached items
        - K larger than n_items is handled gracefully
        - perfect hit rate when all test items are in cache
        - zero hit rate when no test items are in cache
        - partial hit rate computed correctly
        - simulate_requests() before populate() raises ValueError
        - hit_rate is always in [0.0, 1.0]
        - hits + misses == total always
        - result dict contains all required keys

    Integration:
        - full pipeline (score -> populate -> simulate) runs end-to-end
        - larger cache_size_k gives >= hit rate vs smaller cache
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import pytest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]       = "1"

# ----------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------
TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
DIS_MODULE   = os.path.join(PROJECT_ROOT, "05_dissemination")
GNN_MODULE   = os.path.join(PROJECT_ROOT, "03_gnn")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DIS_MODULE)
sys.path.insert(0, GNN_MODULE)

from cache_sim import DemandScorer, CacheSimulator
from lightgcn  import LightGCN


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def make_lightgcn(n_users=10, n_items=20, emb_dim=16, n_layers=2, seed=0):
    """Creates a tiny LightGCN on a small synthetic bipartite graph."""
    rng = np.random.default_rng(seed)
    n_int = 30
    u = rng.integers(0, n_users, size=n_int).astype(np.int64)
    v = (rng.integers(0, n_items, size=n_int) + n_users).astype(np.int64)
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    model = LightGCN(
        n_users=n_users, n_items=n_items,
        emb_dim=emb_dim, n_layers=n_layers,
        edge_src=src, edge_dst=dst,
    )
    model.eval()
    return model


def make_test_df(item_ids):
    """Creates a minimal test DataFrame with the given item_ids."""
    return pd.DataFrame({"user_id": [0] * len(item_ids), "item_id": item_ids})


# ============================================================
# Tests: DemandScorer
# ============================================================

class TestDemandScorer:

    def setup_method(self):
        self.n_users = 10
        self.n_items = 20
        self.model   = make_lightgcn(self.n_users, self.n_items)
        self.scorer  = DemandScorer(self.model)

    def test_output_shape(self):
        """score() must return an array of shape [n_items]."""
        scores = self.scorer.score(list(range(5)))
        assert scores.shape == (self.n_items,), (
            f"Expected shape ({self.n_items},), got {scores.shape}"
        )

    def test_output_dtype(self):
        """score() must return a float32 numpy array."""
        scores = self.scorer.score(list(range(5)))
        assert isinstance(scores, np.ndarray), "Expected np.ndarray"
        assert scores.dtype == np.float32, (
            f"Expected float32, got {scores.dtype}"
        )

    def test_no_nan_or_inf(self):
        """score() must not produce NaN or Inf values."""
        scores = self.scorer.score(list(range(self.n_users)))
        assert not np.isnan(scores).any(), "NaN found in demand scores"
        assert not np.isinf(scores).any(), "Inf found in demand scores"

    def test_scores_vary_across_items(self):
        """Not all items should receive the same score."""
        scores = self.scorer.score(list(range(5)))
        assert scores.std() > 0, (
            "All items received identical scores — model produces constant output"
        )

    def test_different_users_give_different_scores(self):
        """
        Two disjoint user groups should produce different score distributions
        (because their mean embeddings differ).
        """
        group_a = list(range(0, 5))
        group_b = list(range(5, 10))
        scores_a = self.scorer.score(group_a)
        scores_b = self.scorer.score(group_b)
        # Scores don't have to differ for every item, but
        # the arrays as a whole should not be identical
        assert not np.allclose(scores_a, scores_b), (
            "Different user groups produced identical item scores — "
            "mean embedding not varying across users"
        )

    def test_single_user(self):
        """score() must work with a single-user list."""
        scores = self.scorer.score([0])
        assert scores.shape == (self.n_items,)
        assert not np.isnan(scores).any()

    def test_all_users(self):
        """score() must work when all users are included."""
        scores = self.scorer.score(list(range(self.n_users)))
        assert scores.shape == (self.n_items,)

    def test_empty_user_list_raises(self):
        """Empty user list should raise an error (can't take mean of 0 rows)."""
        with pytest.raises(Exception):  # IndexError or RuntimeError
            self.scorer.score([])


# ============================================================
# Tests: CacheSimulator
# ============================================================

class TestCacheSimulator:

    def setup_method(self):
        self.n_items = 20
        self.k       = 5
        self.cache   = CacheSimulator(cache_size_k=self.k)
        # Synthetic scores: item i gets score i (item 19 is highest)
        self.scores  = np.arange(self.n_items, dtype=np.float32)

    # ---- populate() tests ----

    def test_populate_returns_k_items(self):
        """populate() must return exactly K item IDs."""
        top_k = self.cache.populate(0, self.scores)
        assert len(top_k) == self.k, (
            f"Expected {self.k} items, got {len(top_k)}"
        )

    def test_populate_top_k_are_highest_scored(self):
        """Cached items must be the K items with the highest scores."""
        top_k = self.cache.populate(0, self.scores)
        cached_set    = set(top_k.tolist())
        expected_top  = set(range(self.n_items - self.k, self.n_items))  # 15-19
        assert cached_set == expected_top, (
            f"Expected top-K items {expected_top}, got {cached_set}"
        )

    def test_populate_sorted_descending(self):
        """Items in the returned array must be sorted by score descending."""
        top_k = self.cache.populate(0, self.scores)
        scores_of_top = self.scores[top_k]
        assert list(scores_of_top) == sorted(scores_of_top, reverse=True), (
            "Top-K items not sorted by score descending"
        )

    def test_populate_k_larger_than_n_items(self):
        """When K > n_items, populate must return all n_items without crashing."""
        big_cache = CacheSimulator(cache_size_k=1000)
        top_k = big_cache.populate(0, self.scores)
        assert len(top_k) == self.n_items, (
            f"Expected {self.n_items} items (capped), got {len(top_k)}"
        )

    def test_populate_k_equals_one(self):
        """K=1 should return exactly the single highest-scored item."""
        cache = CacheSimulator(cache_size_k=1)
        top_k = cache.populate(0, self.scores)
        assert len(top_k) == 1
        assert top_k[0] == self.n_items - 1, (
            f"K=1 should cache item {self.n_items - 1}, got {top_k[0]}"
        )

    def test_populate_stores_cache_internally(self):
        """After populate(), the node's cache must be retrievable internally."""
        self.cache.populate(0, self.scores)
        assert 0 in self.cache._caches, "Node 0 cache not stored internally"

    # ---- simulate_requests() tests ----

    def test_simulate_before_populate_raises(self):
        """simulate_requests() before populate() must raise ValueError."""
        fresh = CacheSimulator(cache_size_k=5)
        test_df = make_test_df([0, 1, 2])
        with pytest.raises(ValueError):
            fresh.simulate_requests(99, test_df)

    def test_simulate_perfect_hit_rate(self):
        """When all test items are in cache, hit_rate must be 1.0."""
        top_k = self.cache.populate(0, self.scores)  # items 15-19
        test_df = make_test_df(top_k.tolist())        # request exactly those
        result = self.cache.simulate_requests(0, test_df)
        assert result["hit_rate"] == 1.0, (
            f"Expected hit_rate 1.0, got {result['hit_rate']}"
        )

    def test_simulate_zero_hit_rate(self):
        """When no test items are in cache, hit_rate must be 0.0."""
        self.cache.populate(0, self.scores)     # caches items 15-19
        test_df = make_test_df([0, 1, 2, 3])   # items not in cache
        result = self.cache.simulate_requests(0, test_df)
        assert result["hit_rate"] == 0.0, (
            f"Expected hit_rate 0.0, got {result['hit_rate']}"
        )

    def test_simulate_partial_hit_rate(self):
        """Partial overlap must give correct fractional hit_rate."""
        self.cache.populate(0, self.scores)      # caches items 15-19
        # 2 hits (15, 16) and 2 misses (0, 1)
        test_df = make_test_df([15, 16, 0, 1])
        result  = self.cache.simulate_requests(0, test_df)
        assert result["hits"]   == 2
        assert result["misses"] == 2
        assert abs(result["hit_rate"] - 0.5) < 1e-6, (
            f"Expected hit_rate 0.5, got {result['hit_rate']}"
        )

    def test_hits_plus_misses_equals_total(self):
        """hits + misses must always equal total."""
        self.cache.populate(0, self.scores)
        test_df = make_test_df([0, 5, 10, 15, 19])
        result  = self.cache.simulate_requests(0, test_df)
        assert result["hits"] + result["misses"] == result["total"], (
            f"hits({result['hits']}) + misses({result['misses']}) "
            f"!= total({result['total']})"
        )

    def test_hit_rate_in_valid_range(self):
        """hit_rate must be in [0.0, 1.0]."""
        self.cache.populate(0, self.scores)
        test_df = make_test_df(list(range(self.n_items)))
        result  = self.cache.simulate_requests(0, test_df)
        assert 0.0 <= result["hit_rate"] <= 1.0, (
            f"hit_rate {result['hit_rate']} out of range [0, 1]"
        )

    def test_result_dict_has_required_keys(self):
        """Result dict must contain all required keys."""
        self.cache.populate(0, self.scores)
        test_df  = make_test_df([15, 0])
        result   = self.cache.simulate_requests(0, test_df)
        required = {"node_id", "hits", "misses", "total", "hit_rate",
                    "cache_size", "cached_items"}
        missing  = required - set(result.keys())
        assert not missing, f"Missing keys in result: {missing}"

    def test_empty_test_df_gives_zero_total(self):
        """Empty test interactions should give 0 hits, 0 misses, 0.0 hit_rate."""
        self.cache.populate(0, self.scores)
        test_df = make_test_df([])
        result  = self.cache.simulate_requests(0, test_df)
        assert result["total"]    == 0
        assert result["hits"]     == 0
        assert result["hit_rate"] == 0.0

    def test_multiple_nodes_independent(self):
        """Each node must have its own independent cache."""
        scores_0 = np.arange(self.n_items, dtype=np.float32)          # high = 19
        scores_1 = np.arange(self.n_items, dtype=np.float32)[::-1].copy()  # high = 0

        self.cache.populate(0, scores_0)
        self.cache.populate(1, scores_1)

        cached_0 = set(self.cache._caches[0].tolist())
        cached_1 = set(self.cache._caches[1].tolist())

        # Node 0 should cache high-numbered items, node 1 low-numbered
        assert max(cached_0) > min(cached_1), (
            "Node caches are not independent — both cached same items"
        )


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:

    def test_full_pipeline_on_synthetic(self):
        """
        Full pipeline: DemandScorer.score() -> CacheSimulator.populate()
        -> CacheSimulator.simulate_requests() — must run without error
        and produce a valid result dict.
        """
        n_users, n_items = 10, 20
        model   = make_lightgcn(n_users, n_items, seed=5)
        scorer  = DemandScorer(model)
        cache   = CacheSimulator(cache_size_k=5)

        user_ids = list(range(n_users))
        scores   = scorer.score(user_ids)

        top_k    = cache.populate(0, scores)
        test_df  = make_test_df(list(range(n_items)))
        result   = cache.simulate_requests(0, test_df)

        assert result["total"] == n_items
        assert result["hits"] + result["misses"] == result["total"]
        assert 0.0 <= result["hit_rate"] <= 1.0

    def test_larger_cache_gives_higher_or_equal_hit_rate(self):
        """
        Increasing cache size K must never decrease the hit rate.
        (A superset cache can only hit more or equal requests.)
        """
        n_users, n_items = 10, 50
        model   = make_lightgcn(n_users, n_items, seed=7)
        scorer  = DemandScorer(model)

        user_ids = list(range(n_users))
        scores   = scorer.score(user_ids)

        # Simulate test requests for all items
        test_df  = make_test_df(list(range(n_items)))

        results = {}
        for k in [5, 10, 25, 50]:
            c = CacheSimulator(cache_size_k=k)
            c.populate(0, scores)
            r = c.simulate_requests(0, test_df)
            results[k] = r["hit_rate"]

        # Monotonically non-decreasing
        ks = sorted(results.keys())
        for i in range(len(ks) - 1):
            assert results[ks[i]] <= results[ks[i+1]] + 1e-9, (
                f"Hit rate decreased from K={ks[i]} "
                f"({results[ks[i]]:.4f}) to K={ks[i+1]} "
                f"({results[ks[i+1]]:.4f})"
            )

    def test_scores_determine_cache_ranking(self):
        """
        Manually set scores so the highest-scored item is known.
        Verify it appears in the cache.
        """
        n_items = 30
        scores  = np.zeros(n_items, dtype=np.float32)
        scores[7]  = 100.0   # item 7 is the clear top item
        scores[22] = 99.0    # item 22 is second

        cache = CacheSimulator(cache_size_k=3)
        top_k = cache.populate(0, scores)

        assert 7  in top_k.tolist(), "Item 7 (highest score) not in top-K cache"
        assert 22 in top_k.tolist(), "Item 22 (second highest) not in top-K cache"
