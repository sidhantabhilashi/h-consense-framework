"""
test_06_baselines.py
--------------------
Unit tests for Module 6 baselines (LRU, LFU, Flat FedAvg).

Runs WITHOUT real data or disk I/O.
All tests use synthetic DataFrames and tiny in-memory LightGCN models.
Expected runtime: < 15 seconds on CPU.

Usage (run from the code/ directory):
    pytest tests/test_06_baselines.py -v
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

for p in [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "03_gnn"),
    os.path.join(PROJECT_ROOT, "04_federated"),
    os.path.join(PROJECT_ROOT, "05_dissemination"),
    os.path.join(PROJECT_ROOT, "06_baselines"),
]:
    sys.path.insert(0, p)

from lru         import LRUCache
from lfu         import LFUCache
from flat_fedavg import FlatFedAvg
from lightgcn    import LightGCN
from cache_sim   import DemandScorer


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def make_train_df(interactions: list) -> pd.DataFrame:
    """Build a minimal train DataFrame from [(user_id, item_id), ...] pairs."""
    return pd.DataFrame(interactions, columns=["user_id", "item_id"])


def make_test_df(item_ids: list) -> pd.DataFrame:
    return pd.DataFrame({"user_id": [0] * len(item_ids), "item_id": item_ids})


REQUIRED_RESULT_KEYS = {"node_id", "hits", "misses", "total",
                        "hit_rate", "cache_size", "cached_items"}


def make_lightgcn(n_users=10, n_items=20, emb_dim=16, n_layers=2, seed=0):
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


# ============================================================
# Tests: LRUCache
# ============================================================

class TestLRUCache:

    def setup_method(self):
        self.k = 5
        self.cache = LRUCache(cache_size_k=self.k)
        # 10 interactions across items 0-9
        self.train = make_train_df([(i % 3, i) for i in range(10)])

    def test_warm_fills_cache_up_to_k(self):
        """After warm(), internal cache has at most K entries."""
        self.cache.warm(self.train)
        assert len(self.cache._cache) <= self.k

    def test_warm_fills_cache_exactly_k_when_enough_items(self):
        """When > K unique items exist, exactly K are cached."""
        train = make_train_df([(0, i) for i in range(20)])  # 20 unique items
        cache = LRUCache(cache_size_k=5)
        cache.warm(train)
        assert len(cache._cache) == 5

    def test_eviction_keeps_most_recently_used(self):
        """
        After replaying [0,1,2,3,4,5] with K=3:
        cache should contain last 3 distinct items accessed: 3,4,5
        """
        train = make_train_df([(0, i) for i in range(6)])  # items 0-5 in order
        cache = LRUCache(cache_size_k=3)
        cache.warm(train)
        cached = set(cache._cache.keys())
        assert cached == {3, 4, 5}, f"Expected {{3,4,5}}, got {cached}"

    def test_simulate_before_warm_raises(self):
        """simulate_requests() before warm() must raise ValueError."""
        fresh = LRUCache(cache_size_k=5)
        with pytest.raises(ValueError):
            fresh.simulate_requests(0, make_test_df([0, 1]))

    def test_perfect_hit_for_cached_items(self):
        """Items in the cache must register as hits."""
        train = make_train_df([(0, i) for i in range(6)])
        cache = LRUCache(cache_size_k=3)
        cache.warm(train)  # caches items 3, 4, 5
        result = cache.simulate_requests(0, make_test_df([3, 4, 5]))
        assert result["hit_rate"] == 1.0

    def test_zero_hit_for_non_cached_items(self):
        """Items not in cache must register as misses."""
        train = make_train_df([(0, i) for i in range(6)])
        cache = LRUCache(cache_size_k=3)
        cache.warm(train)  # caches 3,4,5
        result = cache.simulate_requests(0, make_test_df([0, 1, 2]))
        assert result["hit_rate"] == 0.0

    def test_hits_plus_misses_equals_total(self):
        self.cache.warm(self.train)
        result = self.cache.simulate_requests(0, make_test_df(list(range(10))))
        assert result["hits"] + result["misses"] == result["total"]

    def test_hit_rate_in_valid_range(self):
        self.cache.warm(self.train)
        result = self.cache.simulate_requests(0, make_test_df(list(range(10))))
        assert 0.0 <= result["hit_rate"] <= 1.0

    def test_result_keys_present(self):
        self.cache.warm(self.train)
        result = self.cache.simulate_requests(0, make_test_df([0, 1]))
        missing = REQUIRED_RESULT_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_k_larger_than_catalogue_no_crash(self):
        """K > unique items in training data must not crash."""
        train = make_train_df([(0, i) for i in range(3)])  # 3 items
        cache = LRUCache(cache_size_k=100)
        cache.warm(train)
        assert len(cache._cache) == 3

    def test_empty_train_df(self):
        """warm() on empty DataFrame must not crash; all test items are misses."""
        cache = LRUCache(cache_size_k=5)
        cache.warm(make_train_df([]))
        result = cache.simulate_requests(0, make_test_df([0, 1, 2]))
        assert result["hit_rate"] == 0.0
        assert result["total"]    == 3

    def test_empty_test_df(self):
        self.cache.warm(self.train)
        result = self.cache.simulate_requests(0, make_test_df([]))
        assert result["total"]    == 0
        assert result["hit_rate"] == 0.0


# ============================================================
# Tests: LFUCache
# ============================================================

class TestLFUCache:

    def setup_method(self):
        self.k = 3
        self.cache = LFUCache(cache_size_k=self.k)
        # Item 0 appears 5x, item 1 appears 3x, item 2 appears 2x, item 3 once
        self.train = make_train_df(
            [(0, 0)] * 5 + [(0, 1)] * 3 + [(0, 2)] * 2 + [(0, 3)] * 1
        )

    def test_warm_selects_top_k_most_frequent(self):
        """Cache must contain the K most frequent items."""
        self.cache.warm(self.train)
        assert set(self.cache._cached_items) == {0, 1, 2}

    def test_warm_exactly_k_items(self):
        """Cache must contain exactly K items when enough unique items exist."""
        self.cache.warm(self.train)
        assert len(self.cache._cached_items) == self.k

    def test_perfect_hit_for_top_items(self):
        """Top-K most frequent items must all be hits."""
        self.cache.warm(self.train)  # caches 0, 1, 2
        result = self.cache.simulate_requests(0, make_test_df([0, 1, 2]))
        assert result["hit_rate"] == 1.0

    def test_zero_hit_for_rare_items(self):
        """Rare items (not in top-K) must be misses."""
        self.cache.warm(self.train)  # caches 0,1,2; item 3 not cached
        result = self.cache.simulate_requests(0, make_test_df([3]))
        assert result["hit_rate"] == 0.0

    def test_simulate_before_warm_raises(self):
        fresh = LFUCache(cache_size_k=5)
        with pytest.raises(ValueError):
            fresh.simulate_requests(0, make_test_df([0]))

    def test_hits_plus_misses_equals_total(self):
        self.cache.warm(self.train)
        result = self.cache.simulate_requests(0, make_test_df([0, 1, 2, 3]))
        assert result["hits"] + result["misses"] == result["total"]

    def test_hit_rate_in_valid_range(self):
        self.cache.warm(self.train)
        result = self.cache.simulate_requests(0, make_test_df([0, 1, 2, 3]))
        assert 0.0 <= result["hit_rate"] <= 1.0

    def test_result_keys_present(self):
        self.cache.warm(self.train)
        result = self.cache.simulate_requests(0, make_test_df([0]))
        missing = REQUIRED_RESULT_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_k_larger_than_unique_items(self):
        """K > unique items must not crash; all unique items cached."""
        train = make_train_df([(0, i) for i in range(3)])
        cache = LFUCache(cache_size_k=100)
        cache.warm(train)
        assert len(cache._cached_items) == 3

    def test_empty_train_df(self):
        cache = LFUCache(cache_size_k=5)
        cache.warm(make_train_df([]))
        result = cache.simulate_requests(0, make_test_df([0, 1]))
        assert result["hit_rate"] == 0.0

    def test_empty_test_df(self):
        self.cache.warm(self.train)
        result = self.cache.simulate_requests(0, make_test_df([]))
        assert result["total"]    == 0
        assert result["hit_rate"] == 0.0

    def test_partial_hit_rate(self):
        """Mixed: 2 cached + 2 uncached -> hit_rate = 0.5"""
        self.cache.warm(self.train)  # caches 0, 1, 2
        result = self.cache.simulate_requests(0, make_test_df([0, 1, 3, 3]))
        assert result["hits"]   == 2
        assert result["misses"] == 2
        assert abs(result["hit_rate"] - 0.5) < 1e-6


# ============================================================
# Tests: FlatFedAvg
# ============================================================

class TestFlatFedAvg:

    def _make_node_data(self, n_nodes=3, n_users=15, n_items=20, seed=0):
        """Synthetic per-node models and train DataFrames."""
        rng = np.random.default_rng(seed)
        node_models = {}
        node_train_dfs = {}
        users_per_node = n_users // n_nodes
        for nid in range(n_nodes):
            start = nid * users_per_node
            end   = start + users_per_node
            model = make_lightgcn(n_users, n_items, seed=nid)
            node_models[nid] = model
            interactions = [
                (u, int(rng.integers(0, n_items)))
                for u in range(start, end)
                for _ in range(5)
            ]
            node_train_dfs[nid] = make_train_df(interactions)
        return node_models, node_train_dfs, n_items

    def test_run_returns_lightgcn(self):
        """run() must return a LightGCN instance."""
        models, trains, n_items = self._make_node_data()
        fl = FlatFedAvg(n_rounds=1, local_epochs=1)
        result = fl.run(models, trains, n_items)
        assert isinstance(result, LightGCN)

    def test_model_parameters_changed_after_training(self):
        """Weights must differ from the random init after 1 round of training."""
        models, trains, n_items = self._make_node_data(seed=42)
        # Snapshot initial weights
        ref_nid   = 0
        init_w    = models[ref_nid].embedding.weight.data.clone()
        fl        = FlatFedAvg(n_rounds=2, local_epochs=2)
        fl.run(models, trains, n_items)
        final_w   = models[ref_nid].embedding.weight.data
        assert not torch.allclose(init_w, final_w), (
            "Model weights unchanged after training — training did not run"
        )

    def test_global_model_same_embedding_shape(self):
        """Embedding shape must be unchanged after aggregation."""
        n_users, n_items = 15, 20
        models, trains, _ = self._make_node_data(n_users=n_users, n_items=n_items)
        orig_shape = models[0].embedding.weight.shape
        fl = FlatFedAvg(n_rounds=1, local_epochs=1)
        global_model = fl.run(models, trains, n_items)
        assert global_model.embedding.weight.shape == orig_shape

    def test_single_round_single_epoch_no_crash(self):
        """Minimal config (1 round, 1 epoch) must not crash."""
        models, trains, n_items = self._make_node_data()
        fl = FlatFedAvg(n_rounds=1, local_epochs=1)
        fl.run(models, trains, n_items)   # should not raise

    def test_scores_from_global_model_are_valid(self):
        """DemandScorer on the returned model must produce finite float32 scores."""
        models, trains, n_items = self._make_node_data()
        fl           = FlatFedAvg(n_rounds=1, local_epochs=1)
        global_model = fl.run(models, trains, n_items)
        scorer       = DemandScorer(global_model)
        scores       = scorer.score(list(range(5)))
        assert scores.dtype == np.float32
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()

    def test_multiple_nodes_aggregate(self):
        """5-node flat FedAvg must complete without error."""
        models, trains, n_items = self._make_node_data(n_nodes=5, n_users=25)
        fl = FlatFedAvg(n_rounds=2, local_epochs=1)
        fl.run(models, trains, n_items)


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:

    def test_lru_and_lfu_hit_rates_in_range(self):
        """Both LRU and LFU must produce hit rates in [0, 1]."""
        train = make_train_df([(0, i) for i in range(20)])
        test  = make_test_df(list(range(20)))
        for Cls in (LRUCache, LFUCache):
            c = Cls(cache_size_k=5)
            c.warm(train)
            r = c.simulate_requests(0, test)
            assert 0.0 <= r["hit_rate"] <= 1.0

    def test_all_baselines_same_result_schema(self):
        """LRU, LFU and (mocked) results must all have the same required keys."""
        train = make_train_df([(0, i) for i in range(10)])
        test  = make_test_df([0, 5, 9])
        for Cls in (LRUCache, LFUCache):
            c = Cls(cache_size_k=5)
            c.warm(train)
            r = c.simulate_requests(0, test)
            missing = REQUIRED_RESULT_KEYS - set(r.keys())
            assert not missing, f"{Cls.__name__} missing keys: {missing}"

    def test_lfu_beats_random(self):
        """
        LFU with K=5 from 20 items must beat random baseline (5/20 = 25%).
        We construct training data so the top-5 items are requested in test too.
        """
        # Top 5 items (0-4) appear many times in train AND test
        train = make_train_df(
            [(0, i) for i in range(5)] * 10   # top 5 very frequent
            + [(0, i) for i in range(5, 20)]   # others once each
        )
        test = make_test_df(list(range(20)))
        cache = LFUCache(cache_size_k=5)
        cache.warm(train)
        result = cache.simulate_requests(0, test)
        random_rate = 5 / 20
        assert result["hit_rate"] >= random_rate, (
            f"LFU hit rate {result['hit_rate']:.4f} below random {random_rate:.4f}"
        )

    def test_larger_k_never_decreases_hit_rate_lfu(self):
        """Increasing K for LFU must give >= hit rate (monotonic)."""
        train = make_train_df([(0, i) for i in range(20)])
        test  = make_test_df(list(range(20)))
        prev  = 0.0
        for k in [2, 5, 10, 20]:
            c = LFUCache(cache_size_k=k)
            c.warm(train)
            r = c.simulate_requests(0, test)
            assert r["hit_rate"] >= prev - 1e-9
            prev = r["hit_rate"]

    def test_larger_k_never_decreases_hit_rate_lru(self):
        """Increasing K for LRU must give >= hit rate (monotonic)."""
        train = make_train_df([(0, i) for i in range(20)])
        test  = make_test_df(list(range(20)))
        prev  = 0.0
        for k in [2, 5, 10, 20]:
            c = LRUCache(cache_size_k=k)
            c.warm(train)
            r = c.simulate_requests(0, test)
            assert r["hit_rate"] >= prev - 1e-9
            prev = r["hit_rate"]
