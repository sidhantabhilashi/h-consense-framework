"""
test_01_data.py
---------------
Unit tests for Module 1 (01_data).

Runs WITHOUT needing the real dataset downloaded.
All functions are tested with small synthetic data so tests
are fast, deterministic, and require no internet.

Usage (run from the code/ directory):
    pytest tests/test_01_data.py -v

What is tested:
    preprocess.py:
        - frequency filtering removes low-count users/items
        - implicit conversion drops rating/timestamp columns
        - id remap produces 0-based contiguous IDs
        - train/test split: no overlap, no empty train sets, total preserved

    graph_builder.py:
        - COO arrays have correct length (2 * n_interactions)
        - item node IDs are correctly offset by n_users
        - no self-loops in bipartite graph
        - all node indices within valid range

    edge_assignment.py:
        - every user assigned to exactly one node
        - assignment is user_id % n_nodes
        - no user appears in two nodes
        - all nodes covered
"""

# Fix macOS OpenMP crash before anything else loads
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import pytest
import pandas as pd
import numpy as np

# ----------------------------------------------------------------
# Path setup:
#   - project root  (for config.py)
#   - 01_data/      (for preprocess, graph_builder, edge_assignment)
# ----------------------------------------------------------------
TESTS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
DATA_MODULE  = os.path.join(PROJECT_ROOT, "01_data")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DATA_MODULE)

# ----------------------------------------------------------------
# Import source modules
# ----------------------------------------------------------------
from preprocess import (
    filter_by_frequency,
    to_implicit,
    remap_ids,
    train_test_split_per_user,
)
from graph_builder import build_bipartite_coo, validate_coo
from edge_assignment import assign_users_to_nodes, compute_edge_stats, validate_assignment


# ============================================================
# Synthetic Data Helper
# ============================================================

def make_synthetic_df(n_users=10, n_items=20, n_ratings=100, seed=0):
    """
    Creates a synthetic ratings DataFrame (user_id, item_id, rating, timestamp).
    Used to test preprocessing functions without real data.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "user_id":   rng.integers(1, n_users + 1, size=n_ratings),
        "item_id":   rng.integers(1, n_items + 1, size=n_ratings),
        "rating":    rng.integers(1, 6,           size=n_ratings),
        "timestamp": rng.integers(800000000, 900000000, size=n_ratings),
    })


# ============================================================
# Tests: preprocess.py — filter_by_frequency
# ============================================================

class TestFilterByFrequency:

    def test_removes_low_freq_users(self):
        """Users with fewer than MIN_USER_RATINGS interactions should be removed."""
        df = pd.DataFrame({
            "user_id":   [1, 1, 1, 1, 1, 2],   # user 2 has only 1 rating
            "item_id":   [1, 2, 3, 4, 5, 1],
            "rating":    [4, 3, 5, 2, 4, 3],
            "timestamp": [0] * 6,
        })
        import config as cfg
        original_u, original_i = cfg.MIN_USER_RATINGS, cfg.MIN_ITEM_RATINGS
        cfg.MIN_USER_RATINGS = 5
        cfg.MIN_ITEM_RATINGS = 1
        result = filter_by_frequency(df)
        cfg.MIN_USER_RATINGS = original_u
        cfg.MIN_ITEM_RATINGS = original_i
        assert 2 not in result["user_id"].values, \
            "User 2 with only 1 rating should have been filtered out."

    def test_keeps_all_when_above_threshold(self):
        """All users/items above threshold should be retained."""
        import config as cfg
        original_u, original_i = cfg.MIN_USER_RATINGS, cfg.MIN_ITEM_RATINGS
        cfg.MIN_USER_RATINGS = 1
        cfg.MIN_ITEM_RATINGS = 1
        df = make_synthetic_df(n_users=5, n_items=5, n_ratings=50)
        result = filter_by_frequency(df)
        cfg.MIN_USER_RATINGS = original_u
        cfg.MIN_ITEM_RATINGS = original_i
        assert len(result) > 0, "Should retain rows when threshold is 1."


# ============================================================
# Tests: preprocess.py — to_implicit
# ============================================================

class TestToImplicit:

    def test_output_has_only_user_item_columns(self):
        df = make_synthetic_df()
        result = to_implicit(df)
        assert list(result.columns) == ["user_id", "item_id"], \
            f"Expected columns [user_id, item_id], got: {list(result.columns)}"

    def test_no_duplicates_in_output(self):
        """Same (user, item) pair should appear only once."""
        df = pd.DataFrame({
            "user_id":   [1, 1, 2],
            "item_id":   [1, 1, 2],   # duplicate (1, 1)
            "rating":    [3, 4, 5],
            "timestamp": [0, 1, 2],
        })
        result = to_implicit(df)
        assert len(result) == 2, \
            f"Duplicate (user, item) pair not removed. Got {len(result)} rows."


# ============================================================
# Tests: preprocess.py — remap_ids
# ============================================================

class TestRemapIds:

    def test_ids_are_zero_based(self):
        df = pd.DataFrame({"user_id": [5, 10, 15], "item_id": [100, 200, 300]})
        result, user_map, item_map = remap_ids(df)
        assert result["user_id"].min() == 0, "Remapped user IDs should start from 0."
        assert result["item_id"].min() == 0, "Remapped item IDs should start from 0."

    def test_ids_are_contiguous(self):
        df = pd.DataFrame({"user_id": [5, 10, 15], "item_id": [100, 200, 300]})
        result, user_map, item_map = remap_ids(df)
        assert sorted(result["user_id"].unique()) == [0, 1, 2], \
            "Remapped user IDs should be contiguous [0, 1, 2]."
        assert sorted(result["item_id"].unique()) == [0, 1, 2], \
            "Remapped item IDs should be contiguous [0, 1, 2]."

    def test_map_length_matches_uniques(self):
        df = make_synthetic_df(n_users=10, n_items=15, n_ratings=60)
        df = to_implicit(df)
        _, user_map, item_map = remap_ids(df)
        assert len(user_map) == df["user_id"].nunique(), \
            "user_map should have one entry per unique user."
        assert len(item_map) == df["item_id"].nunique(), \
            "item_map should have one entry per unique item."


# ============================================================
# Tests: preprocess.py — train_test_split_per_user
# ============================================================

class TestTrainTestSplit:

    def setup_method(self):
        # 3 users, each with 10 unique items
        self.df = pd.DataFrame({
            "user_id": [0] * 10 + [1] * 10 + [2] * 10,
            "item_id": list(range(10)) + list(range(10, 20)) + list(range(20, 30)),
        })

    def test_no_user_has_empty_train(self):
        """Every user must have at least one training item."""
        import config as cfg
        original = cfg.TEST_RATIO
        cfg.TEST_RATIO = 0.2
        train_df, _ = train_test_split_per_user(self.df)
        cfg.TEST_RATIO = original
        for user in self.df["user_id"].unique():
            n_train = (train_df["user_id"] == user).sum()
            assert n_train > 0, \
                f"User {user} has zero training items — split logic bug."

    def test_no_overlap_between_train_and_test(self):
        """Same (user, item) pair must not appear in both train and test."""
        import config as cfg
        original = cfg.TEST_RATIO
        cfg.TEST_RATIO = 0.2
        train_df, test_df = train_test_split_per_user(self.df)
        cfg.TEST_RATIO = original
        train_set = set(zip(train_df["user_id"], train_df["item_id"]))
        test_set  = set(zip(test_df["user_id"],  test_df["item_id"]))
        overlap = train_set & test_set
        assert len(overlap) == 0, \
            f"Found {len(overlap)} (user, item) pairs in BOTH train and test!"

    def test_total_items_preserved(self):
        """Train + test should together cover all original interactions."""
        import config as cfg
        original = cfg.TEST_RATIO
        cfg.TEST_RATIO = 0.2
        train_df, test_df = train_test_split_per_user(self.df)
        cfg.TEST_RATIO = original
        assert len(train_df) + len(test_df) == len(self.df), \
            "Some interactions were lost during train/test split!"


# ============================================================
# Tests: graph_builder.py
# ============================================================

class TestGraphBuilder:

    def setup_method(self):
        self.train_df = pd.DataFrame({
            "user_id": [0, 0, 1, 1, 2],
            "item_id": [0, 1, 1, 2, 0],
        })
        self.n_users = 3
        self.n_items = 3

    def test_coo_length_is_2x_interactions(self):
        """src and dst should each have length 2 * n_interactions."""
        src, dst = build_bipartite_coo(self.train_df, self.n_users)
        expected = 2 * len(self.train_df)
        assert len(src) == expected, \
            f"Expected src length {expected}, got {len(src)}"
        assert len(dst) == expected, \
            f"Expected dst length {expected}, got {len(dst)}"

    def test_item_ids_offset_by_n_users(self):
        """In user->item edges (first half), dst values must be >= n_users."""
        src, dst = build_bipartite_coo(self.train_df, self.n_users)
        n = len(self.train_df)
        dst_u2i = dst[:n]   # first half = user->item direction
        assert (dst_u2i >= self.n_users).all(), \
            "Item node IDs in user->item edges must be >= n_users."

    def test_no_self_loops(self):
        """In a bipartite graph, src[i] should never equal dst[i]."""
        src, dst = build_bipartite_coo(self.train_df, self.n_users)
        self_loops = int((src == dst).sum())
        assert self_loops == 0, \
            f"Found {self_loops} self-loops in bipartite graph!"

    def test_validate_passes_on_valid_input(self):
        """validate_coo should not raise on valid input."""
        src, dst = build_bipartite_coo(self.train_df, self.n_users)
        n_nodes = self.n_users + self.n_items
        validate_coo(src, dst, n_nodes)   # should not raise


# ============================================================
# Tests: edge_assignment.py
# ============================================================

class TestEdgeAssignment:

    def setup_method(self):
        self.train_df = pd.DataFrame({
            "user_id": [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8],
            "item_id": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
        })
        self.n_nodes = 3

    def test_assignment_is_modulo(self):
        """node_id should equal user_id % n_nodes for every user."""
        user_edge_map, _ = assign_users_to_nodes(self.train_df, self.n_nodes)
        for user_id, node_id in user_edge_map.items():
            expected = int(user_id) % self.n_nodes
            assert node_id == expected, \
                f"User {user_id}: expected node {expected}, got {node_id}"

    def test_all_users_assigned(self):
        """Every user in train_df must appear in user_edge_map."""
        user_edge_map, _ = assign_users_to_nodes(self.train_df, self.n_nodes)
        for uid in self.train_df["user_id"].unique():
            assert uid in user_edge_map, \
                f"User {uid} is missing from user_edge_map!"

    def test_no_user_in_two_nodes(self):
        """No user should appear in more than one node."""
        _, node_user_map = assign_users_to_nodes(self.train_df, self.n_nodes)
        all_users = [u for users in node_user_map.values() for u in users]
        assert len(all_users) == len(set(all_users)), \
            "Some users appear in multiple nodes — assignment bug!"

    def test_all_nodes_covered(self):
        """With 9 users mod 3 nodes, all 3 nodes should have users."""
        _, node_user_map = assign_users_to_nodes(self.train_df, self.n_nodes)
        for node_id in range(self.n_nodes):
            assert node_id in node_user_map, \
                f"Node {node_id} has no users assigned!"


# ============================================================
# Run directly (fallback without pytest)
# ============================================================

if __name__ == "__main__":
    pytest.main(["-v", __file__])
