"""
trainer.py
----------
Trains one LightGCN model per edge node using only that node's users.

This simulates federated local training: each edge node trains
independently on its own slice of the interaction data.
No cross-node data sharing happens here.

Training uses BPR loss (Bayesian Personalised Ranking):
    For each user, a positive (rated) item should score higher
    than a randomly sampled negative (unrated) item.

    L_BPR = -sum log sigmoid(y_ui - y_uj) + L2_REG * ||E||^2

Usage:
    python 03_gnn/trainer.py

Inputs:
    data/processed/train.csv
    data/processed/test.csv
    data/processed/stats.json
    data/processed/node_user_map.json
    data/processed/graph_src.npy
    data/processed/graph_dst.npy

Outputs (per node):
    data/models/node_{i}_lightgcn.pt     <- model state_dict
    data/models/node_{i}_meta.json       <- training metadata + metrics
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from lightgcn import LightGCN
from evaluator import evaluate


def log(msg):
    print(f"[trainer] {msg}", flush=True)


# ----------------------------------------------------------------
# Input loading
# ----------------------------------------------------------------

def load_all_inputs():
    """
    Loads all shared data that every node needs.
    Fails fast with a clear message if Module 1 or 2 has not been run.
    """
    required = {
        "train.csv":          os.path.join(config.PROC_DATA_DIR, "train.csv"),
        "test.csv":           os.path.join(config.PROC_DATA_DIR, "test.csv"),
        "stats.json":         os.path.join(config.PROC_DATA_DIR, "stats.json"),
        "node_user_map.json": os.path.join(config.PROC_DATA_DIR, "node_user_map.json"),
        "graph_src.npy":      os.path.join(config.PROC_DATA_DIR, "graph_src.npy"),
        "graph_dst.npy":      os.path.join(config.PROC_DATA_DIR, "graph_dst.npy"),
    }
    for name, path in required.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[ERROR] Missing: {path}\n"
                f"        Run Module 1 first:\n"
                f"          python 01_data/preprocess.py\n"
                f"          python 01_data/graph_builder.py\n"
                f"          python 01_data/edge_assignment.py"
            )

    train_df = pd.read_csv(required["train.csv"])
    test_df  = pd.read_csv(required["test.csv"])

    with open(required["stats.json"]) as f:
        stats = json.load(f)

    with open(required["node_user_map.json"]) as f:
        node_user_map = json.load(f)   # keys are strings

    full_src = np.load(required["graph_src.npy"])
    full_dst = np.load(required["graph_dst.npy"])

    n_users = stats["n_users"]
    n_items = stats["n_items"]

    log(f"Loaded train.csv      : {len(train_df):,} interactions")
    log(f"Loaded test.csv       : {len(test_df):,} interactions")
    log(f"Global graph          : {n_users} users, {n_items} items, "
        f"{len(full_src):,} edges")
    log(f"Edge nodes            : {len(node_user_map)}")

    return train_df, test_df, stats, node_user_map, full_src, full_dst


# ----------------------------------------------------------------
# Per-node graph construction
# ----------------------------------------------------------------

def build_local_coo(train_df, node_user_ids, n_users_global, n_items):
    """
    Builds a local bipartite COO graph for a single edge node.

    Only includes interactions from this node's users, but item IDs
    span the FULL item space [0, n_items) so that the embedding table
    is compatible with global FL aggregation in Module 4.

    User IDs are the GLOBAL remapped IDs (from preprocess.py).
    Item node IDs are offset by n_users_global in the COO arrays.

    Returns:
        local_df   pd.DataFrame  : filtered interactions for this node
        src        np.ndarray    : COO edge sources [2 * n_local_interactions]
        dst        np.ndarray    : COO edge destinations
    """
    local_df = train_df[train_df["user_id"].isin(node_user_ids)].reset_index(drop=True)

    if len(local_df) == 0:
        return local_df, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    user_ids = local_df["user_id"].values.astype(np.int64)
    item_ids = local_df["item_id"].values.astype(np.int64)

    # Offset item IDs by global n_users so they sit in item partition of adj
    item_ids_offset = item_ids + n_users_global

    # Bipartite: user->item and item->user (undirected)
    src = np.concatenate([user_ids, item_ids_offset])
    dst = np.concatenate([item_ids_offset, user_ids])

    return local_df, src, dst


# ----------------------------------------------------------------
# BPR negative sampling
# ----------------------------------------------------------------

def sample_bpr_batch(local_df, n_items, batch_size):
    """
    Samples a batch of (user, pos_item, neg_item) BPR triplets.

    For each sample:
        - user    : randomly chosen user from local interactions
        - pos_item: one of that user's training items
        - neg_item: a random item the user has NOT interacted with

    Negative sampling strategy:
        Sample uniformly at random; retry up to 50 times if the
        sampled item is in the user's positive set. If still not
        found, keep the last sample (acceptable rare collision).

    Returns:
        users     np.ndarray [batch_size]  int64
        pos_items np.ndarray [batch_size]  int64
        neg_items np.ndarray [batch_size]  int64
    """
    # Build user -> set of positive items (for fast negative checking)
    user_pos_items = local_df.groupby("user_id")["item_id"].apply(set).to_dict()
    all_users = list(user_pos_items.keys())

    users     = np.empty(batch_size, dtype=np.int64)
    pos_items = np.empty(batch_size, dtype=np.int64)
    neg_items = np.empty(batch_size, dtype=np.int64)

    rng = np.random.default_rng()

    for idx in range(batch_size):
        uid = all_users[rng.integers(len(all_users))]
        pos = rng.choice(list(user_pos_items[uid]))

        # Sample negative item (not in user's positive set)
        neg = rng.integers(n_items)
        for _ in range(50):             # max 50 retries
            if neg not in user_pos_items[uid]:
                break
            neg = rng.integers(n_items)

        users[idx]     = uid
        pos_items[idx] = pos
        neg_items[idx] = neg

    return users, pos_items, neg_items


# ----------------------------------------------------------------
# BPR loss
# ----------------------------------------------------------------

def bpr_loss(users_emb, items_emb, users, pos_items, neg_items, l2_reg):
    """
    Computes BPR loss for a batch of triplets.

    L = -mean( log sigmoid( score(u,pos) - score(u,neg) ) )
      + l2_reg * ( ||e_u||^2 + ||e_pos||^2 + ||e_neg||^2 ) / batch_size

    Args:
        users_emb  Tensor [n_users, emb_dim]
        items_emb  Tensor [n_items, emb_dim]
        users      LongTensor [B]
        pos_items  LongTensor [B]
        neg_items  LongTensor [B]
        l2_reg     float

    Returns:
        loss  scalar Tensor
    """
    e_u   = users_emb[users]      # [B, emb_dim]
    e_pos = items_emb[pos_items]  # [B, emb_dim]
    e_neg = items_emb[neg_items]  # [B, emb_dim]

    pos_scores = (e_u * e_pos).sum(dim=1)   # [B]
    neg_scores = (e_u * e_neg).sum(dim=1)   # [B]

    bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

    # L2 regularisation on the batch embeddings
    l2 = (e_u.norm(2).pow(2) +
          e_pos.norm(2).pow(2) +
          e_neg.norm(2).pow(2)) / len(users)

    return bpr + l2_reg * l2


# ----------------------------------------------------------------
# Train one node
# ----------------------------------------------------------------

def train_node(node_id, node_user_ids, train_df, test_df,
               n_users, n_items, device):
    """
    Trains a LightGCN model for a single edge node.

    Returns:
        model      LightGCN   : trained model
        meta       dict       : training metadata + final metrics
    """
    log("=" * 56)
    log(f"Training Node {node_id} / {config.NUM_EDGE_NODES - 1}")

    # --- Build local graph ---
    local_train_df, src, dst = build_local_coo(
        train_df, node_user_ids, n_users, n_items
    )

    if len(local_train_df) == 0:
        log(f"  WARNING: Node {node_id} has zero training interactions. Skipping.")
        return None, {"node_id": node_id, "skipped": True, "reason": "no_interactions"}

    n_local_users = local_train_df["user_id"].nunique()
    log(f"  Users on this node  : {len(node_user_ids)}")
    log(f"  Train interactions  : {len(local_train_df):,}")
    log(f"  Unique users in train: {n_local_users}")
    log(f"  Graph edges (bidir) : {len(src):,}")

    # --- Instantiate model ---
    model = LightGCN(
        n_users  = n_users,
        n_items  = n_items,
        emb_dim  = config.EMB_DIM,
        n_layers = config.N_LAYERS,
        edge_src = src,
        edge_dst = dst,
    ).to(device)
    log(f"  Model               : {model}")

    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # --- Training loop ---
    model.train()
    last_loss  = float("inf")
    nan_epochs = 0

    for epoch in range(1, config.N_EPOCHS + 1):
        t0 = time.time()

        # Sample BPR batch
        users_np, pos_np, neg_np = sample_bpr_batch(
            local_train_df, n_items, config.BATCH_SIZE
        )

        users_t    = torch.from_numpy(users_np).long().to(device)
        pos_t      = torch.from_numpy(pos_np).long().to(device)
        neg_t      = torch.from_numpy(neg_np).long().to(device)

        optimizer.zero_grad()
        users_emb, items_emb = model()
        loss = bpr_loss(users_emb, items_emb, users_t, pos_t, neg_t, config.L2_REG)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        # NaN guard
        if np.isnan(loss_val):
            nan_epochs += 1
            log(f"  WARNING: NaN loss at epoch {epoch}. "
                f"Using last good weights. ({nan_epochs} NaN epochs so far)")
            if nan_epochs >= 3:
                log(f"  ERROR: Too many NaN epochs. Stopping training for node {node_id}.")
                break
            continue

        last_loss = loss_val

        if epoch % config.LOG_EVERY == 0 or epoch == config.N_EPOCHS:
            elapsed = time.time() - t0
            log(f"  Epoch {epoch:>3}/{config.N_EPOCHS} | "
                f"Loss: {loss_val:.4f} | "
                f"Time: {elapsed:.2f}s")

    # --- Evaluation ---
    log(f"  Evaluating on test set (Recall@{config.TOP_K}, NDCG@{config.TOP_K})...")
    recall, ndcg = evaluate(
        model, local_train_df, test_df, node_user_ids, k=config.TOP_K
    )
    log(f"  Recall@{config.TOP_K}: {recall:.4f} | NDCG@{config.TOP_K}: {ndcg:.4f}")

    meta = {
        "node_id":       node_id,
        "n_users":       n_users,
        "n_items":       n_items,
        "emb_dim":       config.EMB_DIM,
        "n_layers":      config.N_LAYERS,
        "n_epochs":      config.N_EPOCHS,
        "final_loss":    round(last_loss, 6),
        f"recall@{config.TOP_K}": round(recall, 6),
        f"ndcg@{config.TOP_K}":   round(ndcg, 6),
        "local_train_interactions": len(local_train_df),
        "local_users":   len(node_user_ids),
        "skipped":       False,
    }

    return model, meta


# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------

def save_node_model(model, meta, node_id):
    """Saves model state_dict and metadata JSON."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    model_path = os.path.join(config.MODELS_DIR, f"node_{node_id}_lightgcn.pt")
    meta_path  = os.path.join(config.MODELS_DIR, f"node_{node_id}_meta.json")

    torch.save(model.state_dict(), model_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_kb = os.path.getsize(model_path) / 1024
    log(f"  Saved: {model_path}  ({size_kb:.1f} KB)")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    log("=" * 56)
    log("Module 3 — LightGCN Local Training")
    log("=" * 56)

    # --- Device ---
    device = torch.device("cpu")   # CPU only: stable, no CUDA setup needed
    log(f"Device: {device}")

    # --- Load inputs ---
    train_df, test_df, stats, node_user_map, _, _ = load_all_inputs()
    n_users = stats["n_users"]
    n_items = stats["n_items"]

    # --- Train one model per node ---
    all_meta   = []
    successful = 0

    sorted_node_ids = sorted([int(k) for k in node_user_map.keys()])

    for node_id in sorted_node_ids:
        node_user_ids = node_user_map[str(node_id)]

        model, meta = train_node(
            node_id       = node_id,
            node_user_ids = node_user_ids,
            train_df      = train_df,
            test_df       = test_df,
            n_users       = n_users,
            n_items       = n_items,
            device        = device,
        )

        if model is not None:
            save_node_model(model, meta, node_id)
            successful += 1
        else:
            log(f"  Node {node_id} skipped — no model saved.")

        all_meta.append(meta)

    # --- Summary ---
    log("")
    log("=" * 56)
    log(f"Training summary: {successful}/{len(sorted_node_ids)} nodes trained")
    log("")

    trained_meta = [m for m in all_meta if not m.get("skipped")]
    if trained_meta:
        recalls = [m[f"recall@{config.TOP_K}"] for m in trained_meta]
        ndcgs   = [m[f"ndcg@{config.TOP_K}"]   for m in trained_meta]
        log(f"  Mean Recall@{config.TOP_K} : {np.mean(recalls):.4f}")
        log(f"  Mean NDCG@{config.TOP_K}   : {np.mean(ndcgs):.4f}")
        log(f"  Best Recall@{config.TOP_K} : {np.max(recalls):.4f} (Node {np.argmax(recalls)})")
        log(f"  Worst Recall@{config.TOP_K}: {np.min(recalls):.4f} (Node {np.argmin(recalls)})")

    # Save combined summary
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    log(f"  Saved training summary: {summary_path}")

    log("")
    log("=" * 56)
    log("MODULE 3 COMPLETE — model weights saved for all nodes.")
    log("=" * 56)
    log("Next step: python 04_federated/fl_coordinator.py")


if __name__ == "__main__":
    main()
