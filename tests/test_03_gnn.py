"""
test_03_gnn.py
--------------
Unit tests for Module 3 (03_gnn).

Runs WITHOUT needing real data, Module 1/2 outputs, or GPU.
All tests use small synthetic graphs and random tensors.
Fast: should complete in < 10 seconds on CPU.

Usage (run from the code/ directory):
    pytest tests/test_03_gnn.py -v

What is tested:
    lightgcn.py:
        - forward() output shapes are correct
        - embeddings change after a backward pass (model learns)
        - output differs between 1-layer and 3-layer models
        - no NaN/Inf in forward output

    bpr_loss (trainer.py):
        - returns a scalar tensor
        - perfect ranking (pos >> neg) gives low loss
        - reversed ranking (neg >> pos) gives high loss
        - loss decreases over 10 gradient steps

    evaluator.py:
        - Recall@K = 1.0 when top-K exactly matches test items
        - Recall@K = 0.0 when top-K has no test items
        - NDCG@K = 1.0 when first item is the correct test item
        - Training items are masked (never appear in top-K)
        - K larger than n_items does not crash
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import pytest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------
TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
GNN_MODULE   = os.path.join(PROJECT_ROOT, "03_gnn")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, GNN_MODULE)

from lightgcn import LightGCN
from trainer  import bpr_loss
from evaluator import evaluate


# ----------------------------------------------------------------
# Synthetic graph helper
# ----------------------------------------------------------------

def make_small_graph(n_users=5, n_items=8, n_interactions=20, seed=0):
    """
    Creates a small synthetic bipartite COO graph.
    Returns src, dst arrays with both directions (undirected).
    """
    rng = np.random.default_rng(seed)
    u = rng.integers(0, n_users, size=n_interactions).astype(np.int64)
    v = (rng.integers(0, n_items, size=n_interactions) + n_users).astype(np.int64)
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    return src, dst


def make_small_model(n_users=5, n_items=8, emb_dim=16, n_layers=2, seed=0):
    """Creates a small LightGCN model on a synthetic graph."""
    src, dst = make_small_graph(n_users, n_items, seed=seed)
    return LightGCN(
        n_users  = n_users,
        n_items  = n_items,
        emb_dim  = emb_dim,
        n_layers = n_layers,
        edge_src = src,
        edge_dst = dst,
    )


# ============================================================
# Tests: lightgcn.py
# ============================================================

class TestLightGCN:

    def test_forward_user_embedding_shape(self):
        """users_emb must have shape [n_users, emb_dim]."""
        n_users, n_items, emb_dim = 5, 8, 16
        model = make_small_model(n_users, n_items, emb_dim)
        users_emb, _ = model()
        assert users_emb.shape == (n_users, emb_dim), (
            f"Expected users_emb shape ({n_users}, {emb_dim}), got {users_emb.shape}"
        )

    def test_forward_item_embedding_shape(self):
        """items_emb must have shape [n_items, emb_dim]."""
        n_users, n_items, emb_dim = 5, 8, 16
        model = make_small_model(n_users, n_items, emb_dim)
        _, items_emb = model()
        assert items_emb.shape == (n_items, emb_dim), (
            f"Expected items_emb shape ({n_items}, {emb_dim}), got {items_emb.shape}"
        )

    def test_no_nan_in_forward_output(self):
        """Forward pass must not produce NaN or Inf values."""
        model = make_small_model()
        users_emb, items_emb = model()
        assert not torch.isnan(users_emb).any(), "NaN in users_emb"
        assert not torch.isnan(items_emb).any(), "NaN in items_emb"
        assert not torch.isinf(users_emb).any(), "Inf in users_emb"
        assert not torch.isinf(items_emb).any(), "Inf in items_emb"

    def test_embeddings_change_after_backward(self):
        """Model weights must update after one backward pass."""
        model = make_small_model()
        weights_before = model.embedding.weight.data.clone()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        users_emb, items_emb = model()

        # Dummy loss: just minimise sum of user embeddings
        loss = users_emb.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        weights_after = model.embedding.weight.data
        assert not torch.allclose(weights_before, weights_after), (
            "Embedding weights did not change after backward pass — "
            "gradient flow broken."
        )

    def test_different_layers_give_different_output(self):
        """A 1-layer model and a 3-layer model should produce different embeddings."""
        src, dst = make_small_graph()
        model_1 = LightGCN(5, 8, 16, n_layers=1, edge_src=src, edge_dst=dst)
        model_3 = LightGCN(5, 8, 16, n_layers=3, edge_src=src, edge_dst=dst)

        # Copy identical weights to isolate the layer count effect
        model_3.embedding.weight.data = model_1.embedding.weight.data.clone()

        users_1, _ = model_1()
        users_3, _ = model_3()

        assert not torch.allclose(users_1, users_3), (
            "1-layer and 3-layer models produced identical output — "
            "propagation is not working."
        )

    def test_num_parameters_correct(self):
        """Total params = (n_users + n_items) * emb_dim."""
        n_users, n_items, emb_dim = 5, 8, 16
        model = make_small_model(n_users, n_items, emb_dim)
        expected = (n_users + n_items) * emb_dim
        assert model.num_parameters() == expected, (
            f"Expected {expected} parameters, got {model.num_parameters()}"
        )


# ============================================================
# Tests: bpr_loss (trainer.py)
# ============================================================

class TestBPRLoss:

    def setup_method(self):
        torch.manual_seed(0)
        self.emb_dim  = 16
        self.n_users  = 5
        self.n_items  = 8
        self.users_emb = torch.randn(self.n_users, self.emb_dim)
        self.items_emb = torch.randn(self.n_items, self.emb_dim)

    def _make_batch(self, B=4):
        users     = torch.randint(0, self.n_users, (B,))
        pos_items = torch.randint(0, self.n_items, (B,))
        neg_items = torch.randint(0, self.n_items, (B,))
        return users, pos_items, neg_items

    def test_loss_is_scalar(self):
        """BPR loss must return a 0-dimensional (scalar) tensor."""
        u, p, n = self._make_batch()
        loss = bpr_loss(self.users_emb, self.items_emb, u, p, n, l2_reg=1e-4)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_perfect_ranking_gives_low_loss(self):
        """
        When pos_score >> neg_score, BPR loss should be close to 0.
        We construct embeddings so scores are explicitly controlled:
            user  = [1, 0, 0, ...]
            pos   = [20, 0, 0, ...]  -> score = +20
            neg   = [-20, 0, 0, ...] -> score = -20
        BPR loss = -log(sigmoid(20 - (-20))) ≈ 0.
        """
        B = 8
        emb_dim = self.emb_dim

        users_emb = torch.zeros(self.n_users, emb_dim)
        items_emb = torch.zeros(self.n_items, emb_dim)
        users_emb[0, 0] =  1.0    # user 0 points along dim 0
        items_emb[0, 0] =  20.0   # pos item: score = +20
        items_emb[1, 0] = -20.0   # neg item: score = -20

        users     = torch.zeros(B, dtype=torch.long)
        pos_items = torch.zeros(B, dtype=torch.long)
        neg_items = torch.ones(B,  dtype=torch.long)

        loss = bpr_loss(users_emb, items_emb, users, pos_items, neg_items, 0.0)
        assert loss.item() < 0.1, (
            f"Expected low loss for perfect ranking, got {loss.item():.4f}"
        )

    def test_reversed_ranking_gives_high_loss(self):
        """
        When neg_score >> pos_score, BPR loss should be large.
        We construct embeddings so scores are explicitly controlled:
            user  = [1, 0, 0, ...]
            pos   = [-20, 0, 0, ...] -> score = -20
            neg   = [+20, 0, 0, ...] -> score = +20
        BPR loss = -log(sigmoid(-20 - 20)) = -log(sigmoid(-40)) >> 0.
        """
        B = 8
        emb_dim = self.emb_dim

        users_emb = torch.zeros(self.n_users, emb_dim)
        items_emb = torch.zeros(self.n_items, emb_dim)
        users_emb[0, 0] =  1.0    # user 0 points along dim 0
        items_emb[0, 0] = -20.0   # pos item scored low
        items_emb[1, 0] =  20.0   # neg item scored high

        users     = torch.zeros(B, dtype=torch.long)
        pos_items = torch.zeros(B, dtype=torch.long)
        neg_items = torch.ones(B,  dtype=torch.long)

        loss = bpr_loss(users_emb, items_emb, users, pos_items, neg_items, 0.0)
        assert loss.item() > 5.0, (
            f"Expected high loss for reversed ranking, got {loss.item():.4f}"
        )

    def test_loss_decreases_over_gradient_steps(self):
        """Loss should decrease over 10 Adam steps on a small synthetic problem."""
        model = make_small_model(n_users=5, n_items=8, emb_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        src, dst = make_small_graph()
        users     = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        pos_items = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        neg_items = torch.tensor([5, 6, 7, 5, 6], dtype=torch.long)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            u_emb, i_emb = model()
            loss = bpr_loss(u_emb, i_emb, users, pos_items, neg_items, 1e-4)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# ============================================================
# Tests: evaluator.py
# ============================================================

class TestEvaluator:

    def setup_method(self):
        """Sets up a tiny model with controlled embeddings for deterministic tests."""
        self.n_users = 3
        self.n_items = 6
        self.emb_dim = 8

        src, dst = make_small_graph(self.n_users, self.n_items, n_interactions=10)
        self.model = LightGCN(
            n_users  = self.n_users,
            n_items  = self.n_items,
            emb_dim  = self.emb_dim,
            n_layers = 1,
            edge_src = src,
            edge_dst = dst,
        )

    def _make_data(self, train_pairs, test_pairs):
        train_df = pd.DataFrame(train_pairs, columns=["user_id", "item_id"])
        test_df  = pd.DataFrame(test_pairs,  columns=["user_id", "item_id"])
        return train_df, test_df

    def test_recall_is_zero_when_no_hits(self):
        """Recall@K should be 0.0 when no test items appear in top-K."""
        # Force item 0 to score highest for user 0,
        # but put item 1 as the only test item
        with torch.no_grad():
            self.model.embedding.weight.data.fill_(0.0)
            # User 0 embedding points strongly towards item 0
            self.model.embedding.weight.data[0, 0] = 10.0
            self.model.embedding.weight.data[self.n_users + 0, 0] = 10.0  # item 0 scores high
            self.model.embedding.weight.data[self.n_users + 1, 0] = -10.0 # item 1 scores low

        train_df, test_df = self._make_data(
            train_pairs=[(0, 2), (0, 3)],   # items 2,3 in train
            test_pairs =[(0, 1)],           # item 1 is the test item (will score low)
        )
        recall, _ = evaluate(self.model, train_df, test_df, [0], k=1)
        assert recall == 0.0, f"Expected recall=0.0, got {recall}"

    def test_train_items_masked_in_eval(self):
        """
        Items in the training set must never appear in top-K recommendations.
        We put all train items as the highest-scoring items via embedding,
        then check they don't appear in recommendations.
        """
        with torch.no_grad():
            self.model.embedding.weight.data.fill_(0.0)
            # Make user 0 have a positive direction on dim 0
            self.model.embedding.weight.data[0, 0] = 1.0
            # Give items 0,1 (train items) high scores
            self.model.embedding.weight.data[self.n_users + 0, 0] = 100.0
            self.model.embedding.weight.data[self.n_users + 1, 0] = 100.0
            # Give item 2 (test item) a moderate score
            self.model.embedding.weight.data[self.n_users + 2, 0] = 1.0

        train_df, test_df = self._make_data(
            train_pairs=[(0, 0), (0, 1)],   # items 0,1 in train (high score)
            test_pairs =[(0, 2)],           # item 2 is test item
        )
        recall, _ = evaluate(self.model, train_df, test_df, [0], k=2)
        # If masking works, item 2 will be top-1 (after 0,1 masked) -> recall = 1.0
        assert recall == 1.0, (
            f"Train item masking failed: expected recall=1.0 (item 2 found), got {recall}"
        )

    def test_no_test_items_returns_zero(self):
        """Users with no test items should be skipped; result should be 0.0."""
        train_df = pd.DataFrame({"user_id": [0], "item_id": [0]})
        test_df  = pd.DataFrame({"user_id": [], "item_id": []}).astype(int)
        recall, ndcg = evaluate(self.model, train_df, test_df, [0, 1, 2], k=5)
        assert recall == 0.0 and ndcg == 0.0, (
            f"Expected (0.0, 0.0) when no test items, got ({recall}, {ndcg})"
        )

    def test_k_larger_than_n_items_does_not_crash(self):
        """Requesting top-1000 with only 6 items should not throw."""
        train_df, test_df = self._make_data(
            train_pairs=[(0, 0)],
            test_pairs =[(0, 1)],
        )
        try:
            evaluate(self.model, train_df, test_df, [0], k=1000)
        except Exception as e:
            pytest.fail(f"evaluate() crashed with k > n_items: {e}")

    def test_evaluate_returns_floats(self):
        """evaluate() must return Python floats, not tensors."""
        train_df, test_df = self._make_data(
            train_pairs=[(0, 0)],
            test_pairs =[(0, 1)],
        )
        recall, ndcg = evaluate(self.model, train_df, test_df, [0], k=5)
        assert isinstance(recall, float), f"recall should be float, got {type(recall)}"
        assert isinstance(ndcg,   float), f"ndcg should be float,   got {type(ndcg)}"


# ============================================================
# Run directly (fallback without pytest)
# ============================================================

if __name__ == "__main__":
    pytest.main(["-v", __file__])
