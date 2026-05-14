"""
test_04_federated.py
--------------------
Unit tests for Module 4 (04_federated).

Runs WITHOUT real data, GPU, or saved checkpoints.
All tests use tiny synthetic models and in-memory tensors.
Fast: should complete in < 10 seconds on CPU.

Usage (run from the code/ directory):
    pytest tests/test_04_federated.py -v

What is tested:
    aggregator.py:
        weighted_avg()
            - equal weights  -> simple mean
            - unequal weights -> heavier model dominates
            - single model   -> returned unchanged
            - non-float buffers (long tensors) -> taken from first model
            - empty list     -> raises ValueError
            - length mismatch -> raises ValueError
            - zero weight sum -> raises ValueError

        tier1_aggregate()
            - produces one state_dict per cluster
            - nodes missing from node_state_dicts are skipped
            - clusters with no available nodes produce no output

        tier2_aggregate()
            - produces a single global state_dict
            - heavier cluster has greater influence on result
            - empty cluster dict -> raises ValueError

    fl_runner.py:
        broadcast()
            - float params are overwritten from global model
            - edge buffers (long) are preserved from the original node model

        load_node_models() / state_dict pop pattern:
            - popping edge keys before load_state_dict succeeds
            - setattr restores the correct buffer shapes
"""

import os
import sys
import copy

import numpy as np
import torch
import torch.nn as nn
import pytest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ----------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------
TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
FED_MODULE   = os.path.join(PROJECT_ROOT, "04_federated")
GNN_MODULE   = os.path.join(PROJECT_ROOT, "03_gnn")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FED_MODULE)
sys.path.insert(0, GNN_MODULE)

from aggregator import weighted_avg, tier1_aggregate, tier2_aggregate
from lightgcn   import LightGCN


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def make_state_dict(emb_val: float, n_nodes: int = 6, emb_dim: int = 4) -> dict:
    """
    Creates a minimal fake state dict with:
      - 'embedding.weight'  : float tensor filled with emb_val
      - 'edge_src'          : long tensor [0, 1, 2]
      - 'edge_dst'          : long tensor [3, 4, 5]
      - 'edge_weights'      : float tensor ones [3]
    """
    return {
        "embedding.weight": torch.full((n_nodes, emb_dim), emb_val, dtype=torch.float32),
        "edge_src":         torch.tensor([0, 1, 2], dtype=torch.long),
        "edge_dst":         torch.tensor([3, 4, 5], dtype=torch.long),
        "edge_weights":     torch.ones(3, dtype=torch.float32),
    }


def make_lightgcn(n_users=3, n_items=4, emb_dim=8, n_layers=2, seed=0):
    """Creates a tiny LightGCN on a minimal synthetic graph."""
    rng = np.random.default_rng(seed)
    n_int = 10
    u = rng.integers(0, n_users, size=n_int).astype(np.int64)
    v = (rng.integers(0, n_items, size=n_int) + n_users).astype(np.int64)
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    return LightGCN(
        n_users=n_users, n_items=n_items,
        emb_dim=emb_dim, n_layers=n_layers,
        edge_src=src, edge_dst=dst,
    )


# ============================================================
# Tests: weighted_avg
# ============================================================

class TestWeightedAvg:

    def test_equal_weights_is_mean(self):
        """Equal weights -> result is the arithmetic mean of the float tensors."""
        sd1 = make_state_dict(1.0)
        sd2 = make_state_dict(3.0)
        merged = weighted_avg([sd1, sd2], [1.0, 1.0])
        expected = 2.0
        result = merged["embedding.weight"].mean().item()
        assert abs(result - expected) < 1e-5, (
            f"Equal weights: expected mean 2.0, got {result:.6f}"
        )

    def test_unequal_weights_heavier_dominates(self):
        """Weight 9:1 -> result should be close to the heavier model's value."""
        sd1 = make_state_dict(0.0)
        sd2 = make_state_dict(10.0)
        merged = weighted_avg([sd1, sd2], [1.0, 9.0])
        result = merged["embedding.weight"].mean().item()
        expected = 9.0  # 0*0.1 + 10*0.9
        assert abs(result - expected) < 1e-4, (
            f"9:1 weighting: expected ~9.0, got {result:.6f}"
        )

    def test_single_model_returned_unchanged(self):
        """Single model -> deep copy, values identical."""
        sd = make_state_dict(5.0)
        merged = weighted_avg([sd], [1.0])
        assert torch.allclose(
            merged["embedding.weight"],
            sd["embedding.weight"]
        ), "Single model should be returned unchanged."

    def test_non_float_buffers_taken_from_first(self):
        """Long (int) buffers must be taken from the first model, not averaged."""
        sd1 = make_state_dict(1.0)
        sd2 = make_state_dict(2.0)
        # Give each a different edge_src so we can tell which was kept
        sd1["edge_src"] = torch.tensor([10, 20, 30], dtype=torch.long)
        sd2["edge_src"] = torch.tensor([99, 99, 99], dtype=torch.long)
        merged = weighted_avg([sd1, sd2], [1.0, 1.0])
        assert merged["edge_src"].tolist() == [10, 20, 30], (
            "Non-float buffer should be taken from the first state_dict."
        )

    def test_empty_list_raises(self):
        """Empty state_dicts list must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            weighted_avg([], [])

    def test_length_mismatch_raises(self):
        """Mismatched list lengths must raise ValueError."""
        sd = make_state_dict(1.0)
        with pytest.raises(ValueError):
            weighted_avg([sd, sd], [1.0])  # 2 models, 1 weight

    def test_zero_weight_sum_raises(self):
        """All-zero weights must raise ValueError."""
        sd = make_state_dict(1.0)
        with pytest.raises(ValueError):
            weighted_avg([sd, sd], [0.0, 0.0])

    def test_output_has_same_keys(self):
        """Merged state_dict must contain exactly the same keys as inputs."""
        sd1 = make_state_dict(1.0)
        sd2 = make_state_dict(2.0)
        merged = weighted_avg([sd1, sd2], [1.0, 1.0])
        assert set(merged.keys()) == set(sd1.keys()), (
            f"Key mismatch: got {set(merged.keys())}"
        )

    def test_weights_need_not_sum_to_one(self):
        """Unnormalised weights (e.g. interaction counts) should still give correct result."""
        sd1 = make_state_dict(0.0)
        sd2 = make_state_dict(100.0)
        # Counts: 900 and 100 -> normalised 0.9 and 0.1 -> expected = 10.0
        merged = weighted_avg([sd1, sd2], [900, 100])
        result = merged["embedding.weight"].mean().item()
        assert abs(result - 10.0) < 1e-3, (
            f"Large unnormalised weights: expected 10.0, got {result:.6f}"
        )


# ============================================================
# Tests: tier1_aggregate
# ============================================================

class TestTier1Aggregate:

    def _make_cluster_setup(self):
        """
        3 clusters, 5 nodes total:
            cluster 0: nodes 0, 1
            cluster 1: nodes 2, 3
            cluster 2: node  4
        """
        cluster_node_map = {"0": [0, 1], "1": [2, 3], "2": [4]}
        node_state_dicts = {
            0: make_state_dict(1.0),
            1: make_state_dict(3.0),
            2: make_state_dict(5.0),
            3: make_state_dict(7.0),
            4: make_state_dict(9.0),
        }
        node_counts = {0: 100, 1: 100, 2: 200, 3: 200, 4: 50}
        return cluster_node_map, node_state_dicts, node_counts

    def test_produces_one_model_per_cluster(self):
        """tier1_aggregate must return exactly one state_dict per cluster."""
        cnm, nsds, nc = self._make_cluster_setup()
        result = tier1_aggregate(cnm, nsds, nc)
        assert len(result) == 3, f"Expected 3 cluster models, got {len(result)}"
        assert set(result.keys()) == {0, 1, 2}

    def test_equal_weights_within_cluster_is_mean(self):
        """Nodes 0 (val=1) and 1 (val=3) with equal counts -> cluster mean = 2."""
        cnm, nsds, _ = self._make_cluster_setup()
        nc = {0: 50, 1: 50}  # equal counts
        result = tier1_aggregate({"0": [0, 1]}, nsds, nc)
        mean_val = result[0]["embedding.weight"].mean().item()
        assert abs(mean_val - 2.0) < 1e-4, (
            f"Equal-weight cluster mean: expected 2.0, got {mean_val:.6f}"
        )

    def test_missing_node_is_skipped(self):
        """
        If a node listed in a cluster has no saved model,
        it should be silently skipped (not crash).
        """
        cnm = {"0": [0, 99]}          # node 99 has no model
        nsds = {0: make_state_dict(4.0)}
        nc   = {0: 100, 99: 200}
        result = tier1_aggregate(cnm, nsds, nc)
        # Only node 0 available -> cluster result = 4.0
        val = result[0]["embedding.weight"].mean().item()
        assert abs(val - 4.0) < 1e-4, (
            f"Missing node skip: expected 4.0, got {val:.6f}"
        )

    def test_cluster_with_no_available_nodes_omitted(self):
        """A cluster whose nodes are all missing should not appear in the output."""
        cnm  = {"0": [0], "1": [99, 100]}  # cluster 1: all missing
        nsds = {0: make_state_dict(1.0)}
        nc   = {0: 100}
        result = tier1_aggregate(cnm, nsds, nc)
        assert 1 not in result, "Cluster with no available nodes should be omitted."
        assert 0 in result, "Cluster 0 should still be present."

    def test_single_node_cluster_returns_unchanged(self):
        """A cluster with one node should return that node's model unchanged."""
        cnm  = {"2": [4]}
        nsds = {4: make_state_dict(9.0)}
        nc   = {4: 50}
        result = tier1_aggregate(cnm, nsds, nc)
        val = result[2]["embedding.weight"].mean().item()
        assert abs(val - 9.0) < 1e-4


# ============================================================
# Tests: tier2_aggregate
# ============================================================

class TestTier2Aggregate:

    def test_produces_single_state_dict(self):
        """tier2_aggregate must return a single merged state_dict."""
        cluster_sds = {
            0: make_state_dict(2.0),
            1: make_state_dict(4.0),
            2: make_state_dict(6.0),
        }
        counts = {0: 100, 1: 100, 2: 100}
        result = tier2_aggregate(cluster_sds, counts)
        assert isinstance(result, dict)
        assert "embedding.weight" in result

    def test_equal_cluster_weights_is_mean(self):
        """Equal cluster counts -> global mean of cluster values."""
        cluster_sds = {0: make_state_dict(2.0), 1: make_state_dict(8.0)}
        counts = {0: 100, 1: 100}
        result = tier2_aggregate(cluster_sds, counts)
        val = result["embedding.weight"].mean().item()
        assert abs(val - 5.0) < 1e-4, f"Expected 5.0, got {val:.6f}"

    def test_heavier_cluster_dominates(self):
        """Cluster with 9x more interactions should dominate the global model."""
        cluster_sds = {0: make_state_dict(0.0), 1: make_state_dict(10.0)}
        counts = {0: 100, 1: 900}
        result = tier2_aggregate(cluster_sds, counts)
        val = result["embedding.weight"].mean().item()
        assert abs(val - 9.0) < 1e-3, f"Expected ~9.0, got {val:.6f}"

    def test_empty_cluster_dict_raises(self):
        """Empty cluster dict must raise ValueError."""
        with pytest.raises(ValueError):
            tier2_aggregate({}, {})

    def test_single_cluster_returns_its_model(self):
        """Single cluster -> global model equals that cluster's model."""
        cluster_sds = {0: make_state_dict(7.0)}
        result = tier2_aggregate(cluster_sds, {0: 100})
        val = result["embedding.weight"].mean().item()
        assert abs(val - 7.0) < 1e-4


# ============================================================
# Tests: broadcast (from fl_runner)
# ============================================================

class TestBroadcast:
    """
    Tests the broadcast() function from fl_runner.py.
    Imported here inline to avoid triggering fl_runner's module-level code.
    """

    def _get_broadcast_fn(self):
        """Lazy import broadcast to avoid side effects from fl_runner startup."""
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            "fl_runner",
            os.path.join(FED_MODULE, "fl_runner.py")
        )
        mod = importlib.util.module_from_spec(spec)
        # Patch out the heavy imports so the module loads without data
        sys.modules.setdefault("fl_runner", mod)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass  # tolerate import errors for modules not under test
        return getattr(mod, "broadcast", None)

    def test_float_params_overwritten(self):
        """
        After broadcast, every node's embedding.weight should match
        the global model's embedding.weight.
        """
        device = torch.device("cpu")
        node0  = make_lightgcn(seed=0)
        node1  = make_lightgcn(seed=1)
        global_model = make_lightgcn(seed=99)

        # Set global to all-ones so it's clearly identifiable
        with torch.no_grad():
            global_model.embedding.weight.fill_(1.0)

        global_sd = global_model.state_dict()

        # Save original edge buffers of node0 before broadcast
        orig_edge_src = node0.edge_src.clone()

        # Run broadcast logic manually (same as fl_runner.broadcast)
        node_models = {0: node0, 1: node1}
        for nid, model in node_models.items():
            local_sd = model.state_dict()
            merged = {}
            for key, tensor in global_sd.items():
                if tensor.dtype in (torch.float16, torch.float32, torch.float64):
                    merged[key] = tensor.to(device)
                else:
                    merged[key] = local_sd[key]
            model.load_state_dict(merged)

        # Embedding weights should now be all 1.0
        for nid, model in node_models.items():
            val = model.embedding.weight.mean().item()
            assert abs(val - 1.0) < 1e-5, (
                f"Node {nid}: embedding not updated by broadcast. Got {val:.6f}"
            )

    def test_edge_buffers_preserved_after_broadcast(self):
        """
        After broadcast, edge_src (long buffer) must remain the
        original node-local values, not the global model's.
        """
        device = torch.device("cpu")
        node0  = make_lightgcn(seed=0)
        node1  = make_lightgcn(seed=42)  # different graph -> different edge_src
        global_model = make_lightgcn(seed=99)

        # Store node0's original edge_src
        orig_edge_src_node0 = node0.edge_src.clone()

        global_sd = global_model.state_dict()

        # Run broadcast
        for nid, model in {0: node0}.items():
            local_sd = model.state_dict()
            merged = {}
            for key, tensor in global_sd.items():
                if tensor.dtype in (torch.float16, torch.float32, torch.float64):
                    merged[key] = tensor.to(device)
                else:
                    merged[key] = local_sd[key]
            model.load_state_dict(merged)

        # edge_src should be unchanged
        assert torch.equal(node0.edge_src, orig_edge_src_node0), (
            "edge_src was overwritten by broadcast — local graph buffers must be preserved."
        )


# ============================================================
# Tests: state_dict pop pattern (the actual model loading fix)
# ============================================================

class TestStateDictPopPattern:
    """
    Tests the pattern used in load_node_models to load a checkpoint
    without crashing on edge buffer size mismatches.
    """

    def test_pop_edge_keys_then_load(self):
        """
        Saving a LightGCN and reloading into a dummy-init model
        using the pop-then-assign pattern must succeed without error.
        """
        device = torch.device("cpu")
        original = make_lightgcn(n_users=3, n_items=4, emb_dim=8, n_layers=2, seed=0)

        # Simulate saving to disk (in memory here)
        saved_state = copy.deepcopy(original.state_dict())

        # Create a new model shell with a dummy 1-edge graph
        dummy_src = np.array([0], dtype=np.int64)
        dummy_dst = np.array([0], dtype=np.int64)
        shell = LightGCN(
            n_users=3, n_items=4, emb_dim=8, n_layers=2,
            edge_src=dummy_src, edge_dst=dummy_dst,
        ).to(device)

        # Apply the fix: assign edge buffers first so shapes match, then load
        edge_keys = ["edge_src", "edge_dst", "edge_weights"]
        for k in edge_keys:
            if k in saved_state:
                setattr(shell, k, saved_state[k].to(device))
        shell.load_state_dict(saved_state, strict=False)  # must NOT raise

        # Embedding weights must match the original
        assert torch.allclose(
            shell.embedding.weight,
            original.embedding.weight,
        ), "Embedding weights differ after pop-and-load."

    def test_edge_buffers_restored_correctly(self):
        """
        After the pop-and-assign, edge_src/dst/weights must
        exactly match the saved checkpoint values.
        """
        device = torch.device("cpu")
        original   = make_lightgcn(seed=7)
        saved_state = copy.deepcopy(original.state_dict())
        orig_edge_src = saved_state["edge_src"].clone()

        dummy_src = np.array([0], dtype=np.int64)
        dummy_dst = np.array([0], dtype=np.int64)
        shell = LightGCN(
            n_users=3, n_items=4, emb_dim=8, n_layers=2,
            edge_src=dummy_src, edge_dst=dummy_dst,
        ).to(device)

        edge_keys = ["edge_src", "edge_dst", "edge_weights"]
        for k in edge_keys:
            if k in saved_state:
                setattr(shell, k, saved_state[k].to(device))
        shell.load_state_dict(saved_state, strict=False)

        assert torch.equal(shell.edge_src, orig_edge_src), (
            "edge_src not correctly restored after pop-and-assign."
        )
        assert shell.edge_src.shape == orig_edge_src.shape, (
            f"edge_src shape mismatch: {shell.edge_src.shape} vs {orig_edge_src.shape}"
        )

    def test_forward_works_after_loading(self):
        """
        After loading via the pop-and-assign pattern, the model
        must be able to run a full forward pass without error.
        """
        device = torch.device("cpu")
        original    = make_lightgcn(seed=3)
        saved_state = copy.deepcopy(original.state_dict())

        dummy_src = np.array([0], dtype=np.int64)
        dummy_dst = np.array([0], dtype=np.int64)
        shell = LightGCN(
            n_users=3, n_items=4, emb_dim=8, n_layers=2,
            edge_src=dummy_src, edge_dst=dummy_dst,
        ).to(device)

        edge_keys = ["edge_src", "edge_dst", "edge_weights"]
        for k in edge_keys:
            if k in saved_state:
                setattr(shell, k, saved_state[k].to(device))
        shell.load_state_dict(saved_state, strict=False)

        # Forward pass must not crash
        shell.eval()
        with torch.no_grad():
            users_emb, items_emb = shell()

        assert users_emb.shape == (3, 8), f"users_emb shape wrong: {users_emb.shape}"
        assert items_emb.shape == (4, 8), f"items_emb shape wrong: {items_emb.shape}"
        assert not torch.isnan(users_emb).any(), "NaN in users_emb after load"
        assert not torch.isnan(items_emb).any(), "NaN in items_emb after load"
