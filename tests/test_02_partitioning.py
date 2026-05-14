"""
test_02_partitioning.py
-----------------------
Unit tests for Module 2 (02_partitioning).

Runs WITHOUT needing the real dataset or Module 1 outputs.
All functions are tested with small synthetic similarity matrices so
tests are fast, deterministic, and require no internet.

Usage (run from the code/ directory):
    pytest tests/test_02_partitioning.py -v

What is tested:
    edge_graph.py:
        - jaccard(): identical sets -> 1.0
        - jaccard(): disjoint sets  -> 0.0
        - jaccard(): known overlap  -> correct float
        - jaccard(): empty sets     -> 1.0
        - build_similarity_matrix(): matrix is symmetric
        - build_similarity_matrix(): diagonal is all 1.0
        - build_similarity_matrix(): all values in [0, 1]

    partitioner.py:
        - all nodes assigned (no missing node)
        - exactly NUM_CLUSTERS clusters produced
        - no node appears in two clusters
        - union of all cluster node lists == all node ids
"""

import os
import sys
import numpy as np
import pytest

# ----------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------
TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
PART_MODULE  = os.path.join(PROJECT_ROOT, "02_partitioning")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PART_MODULE)

# ----------------------------------------------------------------
# Imports from source modules
# ----------------------------------------------------------------
from edge_graph import jaccard, build_similarity_matrix
from partitioner import run_spectral_clustering, build_cluster_maps


# ============================================================
# Helpers
# ============================================================

def make_block_diagonal_matrix(n_clusters=3, nodes_per_cluster=3, intra=0.9, inter=0.1):
    """
    Creates a synthetic similarity matrix with clear block structure.
    Nodes within the same cluster have similarity `intra`.
    Nodes in different clusters have similarity `inter`.
    Diagonal is always 1.0.

    This gives spectral clustering a clear signal so tests are deterministic.
    """
    n = n_clusters * nodes_per_cluster
    matrix = np.full((n, n), inter)
    for c in range(n_clusters):
        start = c * nodes_per_cluster
        end   = start + nodes_per_cluster
        matrix[start:end, start:end] = intra
    np.fill_diagonal(matrix, 1.0)
    return matrix


# ============================================================
# Tests: edge_graph.py — jaccard()
# ============================================================

class TestJaccard:

    def test_identical_sets_return_one(self):
        """Jaccard of two identical sets should be 1.0."""
        a = {1, 2, 3, 4, 5}
        assert jaccard(a, a) == 1.0, "Identical sets must give Jaccard = 1.0"

    def test_disjoint_sets_return_zero(self):
        """Jaccard of two completely disjoint sets should be 0.0."""
        a = {1, 2, 3}
        b = {4, 5, 6}
        assert jaccard(a, b) == 0.0, "Disjoint sets must give Jaccard = 0.0"

    def test_partial_overlap_correct_value(self):
        """
        Jaccard({1,2,3}, {2,3,4}) = |{2,3}| / |{1,2,3,4}| = 2/4 = 0.5
        """
        a = {1, 2, 3}
        b = {2, 3, 4}
        result = jaccard(a, b)
        assert abs(result - 0.5) < 1e-10, \
            f"Expected Jaccard = 0.5, got {result}"

    def test_empty_sets_return_one(self):
        """Two empty nodes (no items) are considered identical -> 1.0."""
        assert jaccard(set(), set()) == 1.0, \
            "Two empty sets should return 1.0 (identical empty nodes)"

    def test_one_empty_set_returns_zero(self):
        """One non-empty, one empty: union = non-empty set, intersection = empty -> 0.0."""
        a = {1, 2, 3}
        b = set()
        result = jaccard(a, b)
        assert result == 0.0, \
            f"One empty set should give Jaccard = 0.0, got {result}"


# ============================================================
# Tests: edge_graph.py — build_similarity_matrix()
# ============================================================

class TestBuildSimilarityMatrix:

    def setup_method(self):
        # 4 nodes, each with a known item set
        self.node_item_sets = {
            0: {1, 2, 3},
            1: {2, 3, 4},
            2: {5, 6, 7},
            3: {1, 2, 3},   # identical to node 0
        }
        self.sorted_node_ids = [0, 1, 2, 3]

    def test_matrix_is_symmetric(self):
        matrix = build_similarity_matrix(self.node_item_sets, self.sorted_node_ids)
        diff = np.max(np.abs(matrix - matrix.T))
        assert diff < 1e-10, \
            f"Matrix is not symmetric. Max off-symmetry diff: {diff:.2e}"

    def test_diagonal_is_one(self):
        matrix = build_similarity_matrix(self.node_item_sets, self.sorted_node_ids)
        diag = np.diag(matrix)
        assert np.allclose(diag, 1.0), \
            f"Diagonal should all be 1.0, got: {diag}"

    def test_values_in_range(self):
        matrix = build_similarity_matrix(self.node_item_sets, self.sorted_node_ids)
        assert matrix.min() >= 0.0, \
            f"Matrix values must be >= 0, got min={matrix.min()}"
        assert matrix.max() <= 1.0, \
            f"Matrix values must be <= 1, got max={matrix.max()}"

    def test_identical_nodes_have_sim_one(self):
        """Node 0 and Node 3 have identical item sets -> similarity should be 1.0."""
        matrix = build_similarity_matrix(self.node_item_sets, self.sorted_node_ids)
        # Node 0 is index 0, Node 3 is index 3
        assert abs(matrix[0][3] - 1.0) < 1e-10, \
            f"Identical item sets should give similarity 1.0, got {matrix[0][3]}"

    def test_disjoint_nodes_have_sim_zero(self):
        """Node 0 ({1,2,3}) and Node 2 ({5,6,7}) share no items -> similarity 0.0."""
        matrix = build_similarity_matrix(self.node_item_sets, self.sorted_node_ids)
        # Node 0 = index 0, Node 2 = index 2
        assert abs(matrix[0][2] - 0.0) < 1e-10, \
            f"Disjoint item sets should give similarity 0.0, got {matrix[0][2]}"

    def test_shape_is_correct(self):
        matrix = build_similarity_matrix(self.node_item_sets, self.sorted_node_ids)
        n = len(self.sorted_node_ids)
        assert matrix.shape == (n, n), \
            f"Expected shape ({n},{n}), got {matrix.shape}"


# ============================================================
# Tests: partitioner.py — run_spectral_clustering() + build_cluster_maps()
# ============================================================

class TestPartitioner:

    def setup_method(self):
        # Block-diagonal matrix: 3 clusters of 3 nodes each
        # Intra-cluster similarity = 0.9, inter-cluster = 0.05
        # This gives spectral clustering a very clear signal
        self.n_clusters      = 3
        self.nodes_per_cluster = 3
        self.node_ids        = list(range(9))   # [0..8]
        self.matrix          = make_block_diagonal_matrix(
            n_clusters=self.n_clusters,
            nodes_per_cluster=self.nodes_per_cluster,
            intra=0.9,
            inter=0.05,
        )

    def _get_maps(self):
        labels = run_spectral_clustering(self.matrix, self.n_clusters)
        return build_cluster_maps(labels, self.node_ids)

    def test_all_nodes_assigned(self):
        """Every node in node_ids must appear in cluster_assignments."""
        cluster_assignments, _ = self._get_maps()
        for node_id in self.node_ids:
            assert node_id in cluster_assignments, \
                f"Node {node_id} missing from cluster_assignments!"

    def test_correct_number_of_clusters(self):
        """Exactly n_clusters distinct cluster IDs must be produced."""
        _, cluster_node_map = self._get_maps()
        assert len(cluster_node_map) == self.n_clusters, (
            f"Expected {self.n_clusters} clusters, "
            f"got {len(cluster_node_map)}: {list(cluster_node_map.keys())}"
        )

    def test_no_node_in_two_clusters(self):
        """A node must not appear in more than one cluster."""
        _, cluster_node_map = self._get_maps()
        all_nodes = [n for nodes in cluster_node_map.values() for n in nodes]
        assert len(all_nodes) == len(set(all_nodes)), \
            "Some nodes appear in multiple clusters!"

    def test_union_of_clusters_equals_all_nodes(self):
        """Union of all cluster node lists must equal the full node_ids list."""
        _, cluster_node_map = self._get_maps()
        all_nodes_in_clusters = set(n for nodes in cluster_node_map.values() for n in nodes)
        assert all_nodes_in_clusters == set(self.node_ids), (
            f"Cluster union {all_nodes_in_clusters} != expected {set(self.node_ids)}"
        )

    def test_too_many_clusters_raises(self):
        """Requesting more clusters than nodes should raise ValueError."""
        with pytest.raises(ValueError, match="n_clusters"):
            run_spectral_clustering(self.matrix, n_clusters=100)


# ============================================================
# Run directly (fallback without pytest)
# ============================================================

if __name__ == "__main__":
    pytest.main(["-v", __file__])
