"""
lightgcn.py
-----------
PyTorch implementation of LightGCN for implicit feedback recommendation.

LightGCN strips away feature transformation and non-linear activations,
keeping only neighbourhood aggregation:

    E^(k+1) = A_tilde @ E^(k)

Final embedding is the mean across all K+1 layers (including layer 0):

    e_u = (1 / K+1) * sum_{k=0}^{K} e_u^(k)

Reference:
    He et al. "LightGCN: Simplifying and Powering Graph Convolution
    Network for Recommendation." SIGIR 2020.

NOTE — sparse tensor avoidance:
    PyTorch 2.11 + macOS ARM has a confirmed segfault in
    torch.sparse @ dense matmul when OpenMP worker threads are used.
    This implementation uses index_add_ (edge-wise scatter) instead,
    which is fully dense and thread-safe on all platforms.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """
    LightGCN recommendation model.

    Args:
        n_users   (int)         : number of user nodes
        n_items   (int)         : number of item nodes
        emb_dim   (int)         : embedding dimension
        n_layers  (int)         : number of graph convolution layers
        edge_src  (np.ndarray)  : COO source indices  [2 * n_interactions]
        edge_dst  (np.ndarray)  : COO dest   indices  [2 * n_interactions]

    The graph is a bipartite user-item graph.
    Item node IDs in edge arrays must already be offset by n_users.
    """

    def __init__(self, n_users, n_items, emb_dim, n_layers, edge_src, edge_dst):
        super().__init__()

        self.n_users  = n_users
        self.n_items  = n_items
        self.n_nodes  = n_users + n_items
        self.emb_dim  = emb_dim
        self.n_layers = n_layers

        # Single embedding table for all nodes (users first, then items)
        self.embedding = nn.Embedding(
            num_embeddings = self.n_nodes,
            embedding_dim  = emb_dim,
        )
        nn.init.xavier_uniform_(self.embedding.weight)

        # Store edges as plain long tensors (registered as buffers so they
        # move with .to(device) and are saved with state_dict).
        # We do NOT build a sparse matrix — see module docstring.
        src_t = torch.from_numpy(edge_src.astype(np.int64))  # [E]
        dst_t = torch.from_numpy(edge_dst.astype(np.int64))  # [E]

        # Compute D^{-1/2} normalisation weights per edge: w_e = 1/sqrt(deg_src * deg_dst)
        n = self.n_nodes
        deg = torch.zeros(n, dtype=torch.float32)
        deg.index_add_(0, src_t, torch.ones(len(src_t)))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        edge_weights = deg_inv_sqrt[src_t] * deg_inv_sqrt[dst_t]  # [E]

        self.register_buffer("edge_src",     src_t)
        self.register_buffer("edge_dst",     dst_t)
        self.register_buffer("edge_weights", edge_weights)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _propagate(self, x):
        """
        One LightGCN propagation step using edge-wise scatter.

        For each edge (src -> dst) with weight w:
            out[dst] += w * x[src]

        This replaces sparse @ dense matmul with index_add_, which is
        safe on macOS ARM / PyTorch 2.11 (no OMP thread pool involved).

        Args:
            x  Tensor [n_nodes, emb_dim]
        Returns:
            out Tensor [n_nodes, emb_dim]
        """
        # Gather source embeddings and scale by normalisation weight
        src_emb = x[self.edge_src]                           # [E, D]
        weighted = src_emb * self.edge_weights.unsqueeze(1)  # [E, D]

        # Scatter-add into destination nodes
        out = torch.zeros_like(x)
        out.index_add_(0, self.edge_dst, weighted)
        return out

    def forward(self):
        """
        Runs LightGCN propagation for n_layers steps.

        Returns:
            users_emb  Tensor [n_users, emb_dim]  -- final user embeddings
            items_emb  Tensor [n_items, emb_dim]  -- final item embeddings
        """
        x0 = self.embedding.weight           # [n_nodes, D]
        layer_embs = [x0]

        x = x0
        for _ in range(self.n_layers):
            x = self._propagate(x)
            layer_embs.append(x)

        # Mean pooling across all layers (including layer 0)
        out = torch.stack(layer_embs, dim=0).mean(dim=0)  # [n_nodes, D]

        users_emb = out[:self.n_users]
        items_emb = out[self.n_users:]
        return users_emb, items_emb

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Returns embeddings for a batch of user IDs."""
        users_emb, _ = self()
        return users_emb[user_ids]

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Returns embeddings for a batch of item IDs."""
        _, items_emb = self()
        return items_emb[item_ids]

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"LightGCN(n_users={self.n_users}, n_items={self.n_items}, "
            f"emb_dim={self.emb_dim}, n_layers={self.n_layers}, "
            f"n_edges={len(self.edge_src)}, params={self.num_parameters():,})"
        )
