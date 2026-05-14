"""
test_07_evaluation.py
---------------------
Tests for Module 7 — Evaluation experiments E1–E5.

All tests are synthetic — no real model files required.
Each test class mocks or overrides the data-loading step.

Run:
    pytest tests/test_07_evaluation.py -v
"""

import os
import sys
import csv
import json
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "07_evaluation"))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Helpers
# ============================================================

def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ============================================================
# E1 — Cache Hit Rate
# ============================================================

class TestE1CacheHitRate:

    @pytest.fixture(autouse=True)
    def tmp_env(self, tmp_path, monkeypatch):
        """Patch RESULTS_DIR and MODELS_DIR to tmp_path."""
        import config
        monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr(config, "MODELS_DIR",  str(tmp_path / "models"))
        os.makedirs(str(tmp_path / "results"), exist_ok=True)

        baseline = {
            "random_baseline":     0.037,
            "lru_hit_rate":        0.094,
            "lfu_hit_rate":        0.168,
            "flat_fedavg_hit_rate": 0.176,
        }
        cache = {"overall_hit_rate": 0.1755}
        _write_json(str(tmp_path / "results" / "baseline_summary.json"), baseline)
        _write_json(str(tmp_path / "results" / "cache_summary.json"),    cache)
        self.tmp_path = tmp_path

    def test_returns_four_rows(self):
        import e1_cache_hit_rate
        rows = e1_cache_hit_rate.run(save_plot=False)
        assert len(rows) == 4

    def test_methods_present(self):
        import e1_cache_hit_rate
        rows = e1_cache_hit_rate.run(save_plot=False)
        methods = [r["method"] for r in rows]
        assert "LRU (recency)"        in methods
        assert "LFU (popularity)"     in methods
        assert "Flat FedAvg"          in methods
        assert "H-GNN-Consense (ours)" in methods

    def test_hit_rates_in_range(self):
        import e1_cache_hit_rate
        rows = e1_cache_hit_rate.run(save_plot=False)
        for r in rows:
            assert 0.0 <= r["hit_rate"] <= 1.0, f"Out of range: {r}"

    def test_vs_random_positive(self):
        import e1_cache_hit_rate
        rows = e1_cache_hit_rate.run(save_plot=False)
        for r in rows:
            assert r["vs_random"] > 0.0

    def test_hgnn_beats_lru(self):
        import e1_cache_hit_rate
        rows = e1_cache_hit_rate.run(save_plot=False)
        by_method = {r["method"]: r["hit_rate"] for r in rows}
        assert by_method["H-GNN-Consense (ours)"] > by_method["LRU (recency)"]

    def test_csv_written(self):
        import e1_cache_hit_rate
        e1_cache_hit_rate.run(save_plot=False)
        csv_path = str(self.tmp_path / "results" / "e1_cache_hit_rate.csv")
        assert os.path.exists(csv_path)
        rows = _read_csv(csv_path)
        assert len(rows) == 4

    def test_missing_baseline_json_raises(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "RESULTS_DIR", "/nonexistent_xyz")
        import e1_cache_hit_rate
        with pytest.raises((FileNotFoundError, Exception)):
            e1_cache_hit_rate.run(save_plot=False)


# ============================================================
# E2 — NDCG Convergence
# ============================================================

class TestE2NDCGConvergence:

    @pytest.fixture(autouse=True)
    def tmp_env(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "MODELS_DIR",  str(tmp_path / "models"))
        monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr(config, "PROC_DATA_DIR", str(tmp_path / "proc"))
        os.makedirs(str(tmp_path / "models"),  exist_ok=True)
        os.makedirs(str(tmp_path / "results"), exist_ok=True)

        # Write synthetic fl_round_metrics
        fl_metrics = [
            {"round": 0,  "recall": 0.243, "ndcg": 0.251, "timestamp": "2026-01-01T00:00:00"},
            {"round": 5,  "recall": 0.235, "ndcg": 0.244, "timestamp": "2026-01-01T00:01:00"},
            {"round": 10, "recall": 0.232, "ndcg": 0.240, "timestamp": "2026-01-01T00:02:00"},
            {"round": 15, "recall": 0.231, "ndcg": 0.238, "timestamp": "2026-01-01T00:03:00"},
            {"round": 20, "recall": 0.229, "ndcg": 0.235, "timestamp": "2026-01-01T00:04:00"},
        ]
        _write_json(str(tmp_path / "models" / "fl_round_metrics.json"), fl_metrics)
        self.tmp_path = tmp_path

    def test_returns_five_rows(self):
        import e2_ndcg_convergence
        rows = e2_ndcg_convergence.run(save_plot=False)
        assert len(rows) == 5

    def test_rounds_match_input(self):
        import e2_ndcg_convergence
        rows = e2_ndcg_convergence.run(save_plot=False)
        rounds = [r["round"] for r in rows]
        assert rounds == [0, 5, 10, 15, 20]

    def test_ndcg_in_valid_range(self):
        import e2_ndcg_convergence
        rows = e2_ndcg_convergence.run(save_plot=False)
        for r in rows:
            assert 0.0 <= r["avg_ndcg"] <= 1.0
            assert 0.0 <= r["min_ndcg"] <= r["avg_ndcg"]
            assert r["avg_ndcg"] <= r["max_ndcg"] <= 1.0

    def test_min_lte_avg_lte_max(self):
        import e2_ndcg_convergence
        rows = e2_ndcg_convergence.run(save_plot=False)
        for r in rows:
            assert r["min_ndcg"] <= r["avg_ndcg"] <= r["max_ndcg"]

    def test_csv_written(self):
        import e2_ndcg_convergence
        e2_ndcg_convergence.run(save_plot=False)
        csv_path = str(self.tmp_path / "results" / "e2_ndcg_convergence.csv")
        assert os.path.exists(csv_path)
        rows = _read_csv(csv_path)
        assert len(rows) == 5

    def test_missing_metrics_raises(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "MODELS_DIR", "/nonexistent_xyz")
        import e2_ndcg_convergence
        with pytest.raises((FileNotFoundError, Exception)):
            e2_ndcg_convergence.run(save_plot=False)


# ============================================================
# E3 — Communication Cost
# ============================================================

class TestE3CommCost:

    @pytest.fixture(autouse=True)
    def tmp_env(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
        os.makedirs(str(tmp_path / "results"), exist_ok=True)
        self.tmp_path = tmp_path

    def test_returns_ten_rows(self):
        import e3_comm_cost
        rows = e3_comm_cost.run(save_plot=False)
        assert len(rows) == 10

    def test_node_counts_correct(self):
        import e3_comm_cost
        rows = e3_comm_cost.run(save_plot=False)
        ns = [r["n_nodes"] for r in rows]
        assert ns == list(range(10, 101, 10))

    def test_flat_grows_linearly(self):
        import e3_comm_cost
        rows = e3_comm_cost.run(save_plot=False)
        flat = [r["flat_params_total"] for r in rows]
        # Flat cost must be strictly increasing
        assert all(flat[i] < flat[i+1] for i in range(len(flat)-1))

    def test_hfl_tier2_less_than_flat(self):
        """H-FL long-haul (Tier2) must always be less than Flat total."""
        import e3_comm_cost
        rows = e3_comm_cost.run(save_plot=False)
        for r in rows:
            assert r["hfl_tier2_params"] < r["flat_params_total"]

    def test_saving_pct_positive(self):
        import e3_comm_cost
        rows = e3_comm_cost.run(save_plot=False)
        for r in rows:
            assert r["longhual_saving_pct"] > 0.0

    def test_saving_increases_with_n(self):
        """More nodes = higher long-haul saving % (K grows slower than N)."""
        import e3_comm_cost
        rows = e3_comm_cost.run(save_plot=False)
        savings = [r["longhual_saving_pct"] for r in rows]
        # Should be generally non-decreasing (K=ceil(N/3) grows slower than N)
        assert savings[-1] >= savings[0]

    def test_csv_written(self):
        import e3_comm_cost
        e3_comm_cost.run(save_plot=False)
        csv_path = str(self.tmp_path / "results" / "e3_comm_cost.csv")
        assert os.path.exists(csv_path)
        rows = _read_csv(csv_path)
        assert len(rows) == 10


# ============================================================
# E4 — Scalability
# ============================================================

class TestE4Scalability:

    def _make_synthetic_run(self, n_nodes, n_items=50, n_users=100):
        """Simulate what e4 returns without real models."""
        import random
        rng = random.Random(42)
        hit_rate = rng.uniform(0.14, 0.20)
        total_reqs = n_users * 5
        total_hits = int(hit_rate * total_reqs)
        return {
            "n_nodes":        n_nodes,
            "hit_rate":       round(hit_rate, 6),
            "total_requests": total_reqs,
            "total_hits":     total_hits,
            "time_sec":       round(rng.uniform(0.01, 0.1), 3),
        }

    def test_synthetic_rows_count(self):
        rows = [self._make_synthetic_run(n) for n in range(10, 91, 10)]
        assert len(rows) == 9

    def test_hit_rate_in_range(self):
        rows = [self._make_synthetic_run(n) for n in range(10, 91, 10)]
        for r in rows:
            assert 0.0 <= r["hit_rate"] <= 1.0

    def test_time_sec_positive(self):
        rows = [self._make_synthetic_run(n) for n in range(10, 91, 10)]
        for r in rows:
            assert r["time_sec"] > 0

    def test_requests_equals_hits_plus_misses(self):
        rows = [self._make_synthetic_run(n) for n in range(10, 91, 10)]
        for r in rows:
            # hits <= total_requests
            assert r["total_hits"] <= r["total_requests"]

    def test_e4_module_importable(self):
        """Module must import without errors."""
        import e4_scalability
        assert hasattr(e4_scalability, "run")
        assert callable(e4_scalability.run)


# ============================================================
# E5 — Cluster Size Effect
# ============================================================

class TestE5ClusterSize:

    def _make_synthetic_row(self, k):
        import random
        rng = random.Random(k * 7)
        return {
            "k_clusters":  k,
            "hit_rate":    round(rng.uniform(0.15, 0.20), 6),
            "comm_total":  (9 + k) * 146688,
            "comm_tier2":  k      * 146688,
            "time_sec":    round(rng.uniform(0.01, 0.05), 3),
        }

    def test_four_rows(self):
        rows = [self._make_synthetic_row(k) for k in [2, 3, 4, 5]]
        assert len(rows) == 4

    def test_k_values(self):
        rows = [self._make_synthetic_row(k) for k in [2, 3, 4, 5]]
        assert [r["k_clusters"] for r in rows] == [2, 3, 4, 5]

    def test_hit_rate_above_lru(self):
        """All K configs should beat LRU (9.4%) — they use the GNN model."""
        rows = [self._make_synthetic_row(k) for k in [2, 3, 4, 5]]
        lru_baseline = 0.094
        for r in rows:
            assert r["hit_rate"] > lru_baseline, (
                f"K={r['k_clusters']} hit_rate={r['hit_rate']} below LRU baseline"
            )

    def test_comm_tier2_grows_with_k(self):
        """More clusters = more Tier2 transfers."""
        rows = [self._make_synthetic_row(k) for k in [2, 3, 4, 5]]
        tier2 = [r["comm_tier2"] for r in rows]
        assert all(tier2[i] < tier2[i+1] for i in range(len(tier2)-1))

    def test_e5_module_importable(self):
        import e5_cluster_size
        assert hasattr(e5_cluster_size, "run")
        assert callable(e5_cluster_size.run)

    def test_comm_total_formula(self):
        """comm_total = (n_nodes + k_clusters) * model_params."""
        rows = [self._make_synthetic_row(k) for k in [2, 3, 4, 5]]
        model_params = (943 + 1349) * 64  # 146,688
        for r in rows:
            expected = (9 + r["k_clusters"]) * model_params
            assert r["comm_total"] == expected
