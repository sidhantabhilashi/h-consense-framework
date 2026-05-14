"""
aggregator.py
-------------
Pure aggregation functions for Hierarchical Federated Averaging (H-FedAvg).

Two-tier structure:
    Tier-1 : within-cluster aggregation  (node models  -> cluster model)
    Tier-2 : global aggregation          (cluster models -> global model)

Both tiers use the same weighted_avg() function — the only difference is
what gets passed in (node state_dicts vs cluster state_dicts).

Weighting formula:
    w_i = n_interactions_i / sum(n_interactions)

This file contains ONLY pure functions.
No file I/O. No model instantiation. No training.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import torch


# ----------------------------------------------------------------
# Core weighted averaging
# ----------------------------------------------------------------

def weighted_avg(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    Compute a weighted average of model state_dicts.

    Args:
        state_dicts : list of model.state_dict() outputs
        weights     : relative weights (need not sum to 1 — normalised here)

    Returns:
        A single merged state_dict with the same keys.

    Raises:
        ValueError if lists are empty or lengths differ.
    """
    if not state_dicts:
        raise ValueError("weighted_avg: state_dicts list is empty.")
    if len(state_dicts) != len(weights):
        raise ValueError(
            f"weighted_avg: {len(state_dicts)} state_dicts but "
            f"{len(weights)} weights."
        )
    if len(state_dicts) == 1:
        # Single model — return a deep copy, no averaging needed
        return copy.deepcopy(state_dicts[0])

    total = sum(weights)
    if total <= 0:
        raise ValueError(f"weighted_avg: weight sum is {total} (must be > 0).")
    norm_w = [w / total for w in weights]

    # Reference shape from first model to detect node-local buffers
    ref_shapes = {key: tensor.shape for key, tensor in state_dicts[0].items()}

    def _is_averageable(key, tensor):
        """Only average float tensors whose shape is identical across ALL models."""
        if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
            return False
        # Check every model has the same shape for this key
        for sd in state_dicts[1:]:
            if sd[key].shape != ref_shapes[key]:
                return False
        return True

    # Initialise output
    merged = {}
    for key, tensor in state_dicts[0].items():
        if _is_averageable(key, tensor):
            merged[key] = torch.zeros_like(tensor, dtype=torch.float32)
        else:
            # Non-float or node-local buffers (edge_src, edge_dst,
            # edge_weights) — take from the first model unchanged
            merged[key] = tensor.clone()

    # Accumulate weighted sum for averageable params only
    for sd, w in zip(state_dicts, norm_w):
        for key, tensor in sd.items():
            if _is_averageable(key, tensor):
                merged[key] += tensor.to(torch.float32) * w

    return merged


# ----------------------------------------------------------------
# Tier-1: within-cluster aggregation
# ----------------------------------------------------------------

def tier1_aggregate(
    cluster_node_map: Dict[str, List[int]],
    node_state_dicts: Dict[int, Dict[str, torch.Tensor]],
    node_interaction_counts: Dict[int, int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Aggregate node models within each cluster (Tier-1).

    Args:
        cluster_node_map        : {cluster_id_str: [node_id, ...]}
                                  e.g. {"0": [0,2,6], "1": [1,3,4,5], "2": [7,8]}
        node_state_dicts        : {node_id: state_dict}
        node_interaction_counts : {node_id: n_train_interactions}

    Returns:
        {cluster_id_int: aggregated_state_dict}
    """
    cluster_models: Dict[int, Dict[str, torch.Tensor]] = {}

    for cluster_id_str, node_ids in cluster_node_map.items():
        cluster_id = int(cluster_id_str)

        # Filter to nodes that actually have a trained model
        available = [nid for nid in node_ids if nid in node_state_dicts]
        if not available:
            continue

        sds      = [node_state_dicts[nid] for nid in available]
        counts   = [node_interaction_counts.get(nid, 1) for nid in available]

        cluster_models[cluster_id] = weighted_avg(sds, counts)

    return cluster_models


# ----------------------------------------------------------------
# Tier-2: global aggregation
# ----------------------------------------------------------------

def tier2_aggregate(
    cluster_state_dicts: Dict[int, Dict[str, torch.Tensor]],
    cluster_interaction_counts: Dict[int, int],
) -> Dict[str, torch.Tensor]:
    """
    Aggregate cluster models into a single global model (Tier-2).

    Args:
        cluster_state_dicts         : {cluster_id: state_dict}
        cluster_interaction_counts  : {cluster_id: total n_interactions}

    Returns:
        global state_dict
    """
    if not cluster_state_dicts:
        raise ValueError("tier2_aggregate: no cluster models provided.")

    cluster_ids = sorted(cluster_state_dicts.keys())
    sds         = [cluster_state_dicts[cid] for cid in cluster_ids]
    counts      = [cluster_interaction_counts.get(cid, 1) for cid in cluster_ids]

    return weighted_avg(sds, counts)
