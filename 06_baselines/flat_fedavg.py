"""
flat_fedavg.py
--------------
Flat FedAvg baseline for H-GNN-Consense.

Standard FedAvg with NO hierarchical clustering:
    - All 9 nodes aggregate directly into one global model each round
    - Same LightGCN architecture, same local_epochs, same n_rounds as
      Module 4 H-FedAvg (config.FL_ROUNDS, config.LOCAL_EPOCHS)
    - After training, the global model is used by DemandScorer + CacheSimulator
      (identical to Module 5) to produce a hit rate for comparison.

This isolates the benefit of H-GNN-Consense's two-tier hierarchy:
    Flat FedAvg vs H-FedAvg = effect of hierarchical clustering.
"""

from __future__ import annotations

import os
import sys
import copy
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim


class FlatFedAvg:
    """
    Standard single-tier FedAvg.

    Each round:
        1. Every node fine-tunes locally for `local_epochs` epochs
        2. All node state_dicts are averaged (weighted by n_interactions)
        3. Global model is broadcast back to all nodes
    """

    def __init__(
        self,
        n_rounds:     int   = 20,
        local_epochs: int   = 5,
        lr:           float = 1e-3,
        device:       str   = "cpu",
    ):
        self.n_rounds     = n_rounds
        self.local_epochs = local_epochs
        self.lr           = lr
        self.device       = torch.device(device)

    # ------------------------------------------------------------------
    # Internal helpers (mirror fl_runner.py patterns)
    # ------------------------------------------------------------------

    def _local_finetune(
        self,
        model:      "LightGCN",
        train_df:   pd.DataFrame,
        n_items:    int,
    ) -> int:
        """Fine-tune `model` on `train_df` for `local_epochs` epochs.
        Returns number of training interactions used (for weighting)."""
        from trainer import bpr_loss, build_local_coo, sample_bpr_batch

        if train_df.empty:
            return 0

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        model.train()

        for _ in range(self.local_epochs):
            optimizer.zero_grad()
            u_emb, i_emb = model()
            users_np, pos_np, neg_np = sample_bpr_batch(train_df, n_items, 512)
            users_t = torch.tensor(users_np, dtype=torch.long).to(self.device)
            pos_t   = torch.tensor(pos_np,   dtype=torch.long).to(self.device)
            neg_t   = torch.tensor(neg_np,   dtype=torch.long).to(self.device)

            loss = bpr_loss(u_emb, i_emb, users_t, pos_t, neg_t, 1e-4)
            loss.backward()
            optimizer.step()

        model.eval()
        return len(train_df)

    def _weighted_avg(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        weights:     List[float],
    ) -> Dict[str, torch.Tensor]:
        """Weighted average of state_dicts; skip node-local edge buffers."""
        if len(state_dicts) == 1:
            return copy.deepcopy(state_dicts[0])

        total  = sum(weights)
        norm_w = [w / total for w in weights]
        ref_shapes = {k: v.shape for k, v in state_dicts[0].items()}

        def _avg_ok(key, tensor):
            if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
                return False
            for sd in state_dicts[1:]:
                if sd[key].shape != ref_shapes[key]:
                    return False
            return True

        merged = {}
        for key, tensor in state_dicts[0].items():
            if _avg_ok(key, tensor):
                merged[key] = torch.zeros_like(tensor, dtype=torch.float32)
            else:
                merged[key] = tensor.clone()

        for sd, w in zip(state_dicts, norm_w):
            for key, tensor in sd.items():
                if _avg_ok(key, tensor):
                    merged[key] += tensor.to(torch.float32) * w

        return merged

    def _broadcast(
        self,
        node_models: Dict[int, "LightGCN"],
        global_sd:   Dict[str, torch.Tensor],
    ) -> None:
        """Load global weights into every node model (preserve edge buffers)."""
        edge_keys = {"edge_src", "edge_dst", "edge_weights"}
        for nid, model in node_models.items():
            local_sd = model.state_dict()
            merged   = {}
            for key, tensor in global_sd.items():
                merged[key] = local_sd[key] if key in edge_keys else tensor.to(self.device)
            for k in edge_keys:
                if k in merged:
                    setattr(model, k, merged[k])
            model.load_state_dict(merged, strict=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        node_models:    Dict[int, "LightGCN"],
        node_train_dfs: Dict[int, pd.DataFrame],
        n_items:        int,
        log_fn          = None,
    ) -> "LightGCN":
        """
        Run flat FedAvg for `n_rounds` rounds.

        Args:
            node_models    : dict {node_id: LightGCN} — pre-initialised models
                             (loaded from node_{i}_lightgcn.pt checkpoints)
            node_train_dfs : dict {node_id: DataFrame} — per-node training data
            n_items        : total number of items (for negative sampling)
            log_fn         : optional callable(str) for progress logging

        Returns:
            The global LightGCN model after all rounds.
        """
        _log = log_fn if log_fn else (lambda m: None)

        # Identify the first node to clone as global template
        ref_nid   = next(iter(node_models))
        global_sd = copy.deepcopy(node_models[ref_nid].state_dict())

        for round_idx in range(1, self.n_rounds + 1):
            _log(f"  Flat FedAvg round {round_idx}/{self.n_rounds}")

            # Step 1: local fine-tune
            state_dicts: List[Dict] = []
            counts:      List[float] = []

            for nid, model in node_models.items():
                n = self._local_finetune(model, node_train_dfs[nid], n_items)
                state_dicts.append(model.state_dict())
                counts.append(max(n, 1))

            # Step 2: flat aggregate — ALL nodes at once
            global_sd = self._weighted_avg(state_dicts, counts)

            # Step 3: broadcast
            self._broadcast(node_models, global_sd)

        # Return global model (use first node as carrier)
        global_model = node_models[ref_nid]
        global_model.load_state_dict(global_sd, strict=False)
        global_model.eval()
        return global_model
