"""
evaluator.py
------------
Evaluation metrics for LightGCN: Recall@K and NDCG@K.

Evaluation protocol:
    For each user in the test set:
        1. Score all items using the trained model.
        2. Mask out items the user already interacted with in the training set.
        3. Rank the remaining items by score (descending).
        4. Compute Recall@K and NDCG@K against the user's test items.

    Recall@K  = fraction of test items that appear in the top-K recommendations.
    NDCG@K    = normalised discounted cumulative gain — rewards hitting
                test items earlier in the ranked list.

This file contains ONLY metric computation.
No training. No file I/O. No knowledge of nodes or clusters.

Usage (imported by trainer.py):
    from evaluator import evaluate
    recall, ndcg = evaluate(model, train_df, test_df, user_ids, k=10)
"""

import numpy as np
import torch


def _dcg_at_k(hits, k):
    """
    Computes Discounted Cumulative Gain for a single user.

    hits  : np.array of 0/1, length k — 1 if position i is a test item
    k     : cutoff

    DCG@K = sum_{i=1}^{K} hits[i] / log2(i + 1)
    """
    hits = hits[:k].astype(np.float32)
    positions = np.arange(1, len(hits) + 1, dtype=np.float32)
    discounts = np.log2(positions + 1)
    return float(np.sum(hits / discounts))


def _idcg_at_k(n_relevant, k):
    """
    Ideal DCG: best possible DCG when all relevant items are ranked first.

    n_relevant : number of test items for this user
    k          : cutoff
    """
    ideal_hits = min(n_relevant, k)
    positions  = np.arange(1, ideal_hits + 1, dtype=np.float32)
    return float(np.sum(1.0 / np.log2(positions + 1)))


def evaluate(model, train_df, test_df, user_ids, k=10):
    """
    Evaluates a LightGCN model on held-out test interactions.

    Args:
        model      LightGCN      : trained model (in eval mode)
        train_df   pd.DataFrame  : columns [user_id, item_id] — training interactions
                                   used to MASK already-seen items
        test_df    pd.DataFrame  : columns [user_id, item_id] — held-out interactions
        user_ids   list[int]     : user IDs to evaluate (typically the node's users)
        k          int           : recommendation cutoff (default 10)

    Returns:
        recall  float : mean Recall@K across all evaluated users
        ndcg    float : mean NDCG@K   across all evaluated users

    Users with no test items are skipped (not counted in mean).
    """
    model.eval()

    with torch.no_grad():
        users_emb, items_emb = model()
        # Score matrix: [n_users, n_items]
        # We'll index into this per-user so compute full matrix once
        all_scores = torch.matmul(users_emb, items_emb.T).cpu().numpy()
        # all_scores[u, i] = dot product of user u and item i embeddings

    n_items = model.n_items

    # Build lookup dicts for fast per-user access
    train_items_per_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    test_items_per_user  = test_df.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls = []
    ndcgs   = []

    for uid in user_ids:
        test_items  = test_items_per_user.get(uid, set())
        train_items = train_items_per_user.get(uid, set())

        # Skip users with no test items (nothing to evaluate)
        if len(test_items) == 0:
            continue

        # Get this user's scores over all items
        scores = all_scores[uid].copy()        # [n_items]

        # Mask out training items by setting their score to -inf
        # so they never appear in top-K recommendations
        for item_id in train_items:
            if item_id < n_items:              # guard against out-of-range
                scores[item_id] = -np.inf

        # Rank items by score (descending) and take top-K
        top_k_items = np.argsort(scores)[::-1][:k]

        # Compute Recall@K
        n_hits = len(set(top_k_items) & test_items)
        recall = n_hits / min(len(test_items), k)
        recalls.append(recall)

        # Compute NDCG@K
        hits_array = np.array(
            [1 if item in test_items else 0 for item in top_k_items],
            dtype=np.float32
        )
        dcg  = _dcg_at_k(hits_array, k)
        idcg = _idcg_at_k(len(test_items), k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    if len(recalls) == 0:
        return 0.0, 0.0

    return float(np.mean(recalls)), float(np.mean(ndcgs))
