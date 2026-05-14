"""
fl_runner.py
------------
Orchestrates H-FedAvg federated learning across all edge nodes.

Pipeline per FL round:
    1. (Optional) Local fine-tune  — each node trains for LOCAL_EPOCHS
    2. Tier-1 aggregate            — node models -> cluster models
    3. Tier-2 aggregate            — cluster models -> global model
    4. Broadcast                   — all nodes receive global model weights
    5. Evaluate (every EVAL_EVERY) — mean Recall@10 + NDCG@10 across nodes

Inputs  (from Modules 1-3):
    data/processed/stats.json
    data/processed/cluster_node_map.json
    data/processed/node_user_map.json
    data/processed/train.csv
    data/processed/test.csv
    data/processed/graph_src.npy
    data/processed/graph_dst.npy
    data/models/node_{i}_lightgcn.pt
    data/models/node_{i}_meta.json

Outputs:
    data/models/global_lightgcn.pt
    data/models/fl_round_metrics.json
    data/models/fl_summary.json

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \\
    python 04_federated/fl_runner.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import json
import time
import copy
from datetime import datetime, timezone

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# ----------------------------------------------------------------
# Path setup — allow running from project root or 04_federated/
# ----------------------------------------------------------------
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "03_gnn"))
sys.path.insert(0, THIS_DIR)

import config
from lightgcn   import LightGCN
from trainer    import bpr_loss, build_local_coo, sample_bpr_batch
from evaluator  import evaluate
from aggregator import tier1_aggregate, tier2_aggregate


# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------

def log(msg: str):
    print(f"[fl_runner] {msg}", flush=True)


# ----------------------------------------------------------------
# Load all inputs
# ----------------------------------------------------------------

def load_inputs():
    """Loads all required files. Fails fast with clear messages."""
    required = {
        "stats.json":          os.path.join(config.PROC_DATA_DIR, "stats.json"),
        "cluster_node_map":    os.path.join(config.PROC_DATA_DIR, "cluster_node_map.json"),
        "node_user_map":       os.path.join(config.PROC_DATA_DIR, "node_user_map.json"),
        "train.csv":           os.path.join(config.PROC_DATA_DIR, "train.csv"),
        "test.csv":            os.path.join(config.PROC_DATA_DIR, "test.csv"),
        "graph_src.npy":       os.path.join(config.PROC_DATA_DIR, "graph_src.npy"),
        "graph_dst.npy":       os.path.join(config.PROC_DATA_DIR, "graph_dst.npy"),
    }
    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                f"Run Modules 1, 2, and 3 first."
            )

    with open(required["stats.json"]) as f:
        stats = json.load(f)
    with open(required["cluster_node_map"]) as f:
        cluster_node_map = json.load(f)   # {str: [int,...]}
    with open(required["node_user_map"]) as f:
        node_user_map = json.load(f)       # {str: [int,...]}

    train_df = pd.read_csv(required["train.csv"])
    test_df  = pd.read_csv(required["test.csv"])
    full_src = np.load(required["graph_src.npy"])
    full_dst = np.load(required["graph_dst.npy"])

    return stats, cluster_node_map, node_user_map, train_df, test_df, full_src, full_dst


def load_node_models(node_ids, n_users, n_items, device):
    """
    Loads saved LightGCN models and metadata for all nodes.

    Returns:
        node_models : {node_id: LightGCN}
        node_meta   : {node_id: dict}  — includes local_train_interactions
    """
    node_models = {}
    node_meta   = {}

    for nid in node_ids:
        pt_path   = os.path.join(config.MODELS_DIR, f"node_{nid}_lightgcn.pt")
        meta_path = os.path.join(config.MODELS_DIR, f"node_{nid}_meta.json")

        if not os.path.exists(pt_path):
            log(f"  WARNING: node_{nid}_lightgcn.pt not found — skipping node {nid}")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        node_meta[nid] = meta

        # Rebuild model shell with same architecture
        # Edges are stored inside the .pt state_dict as buffers —
        # we need a dummy graph to init the model shell, then
        # load_state_dict overwrites everything including edge buffers.
        dummy_src = np.array([0], dtype=np.int64)
        dummy_dst = np.array([0], dtype=np.int64)
        model = LightGCN(
            n_users  = n_users,
            n_items  = n_items,
            emb_dim  = meta["emb_dim"],
            n_layers = meta["n_layers"],
            edge_src = dummy_src,
            edge_dst = dummy_dst,
        ).to(device)

        state = torch.load(pt_path, map_location=device, weights_only=True)
        # edge buffers are registered buffers with node-specific sizes.
        # Assign them FIRST so the shell has the right shapes, then
        # load the rest with strict=False (edge keys still in state dict
        # but already match after pre-assignment).
        edge_keys = ["edge_src", "edge_dst", "edge_weights"]
        for k in edge_keys:
            if k in state:
                setattr(model, k, state[k].to(device))
        model.load_state_dict(state, strict=False)
        model.eval()
        node_models[nid] = model

    log(f"Loaded {len(node_models)} node models.")
    return node_models, node_meta


# ----------------------------------------------------------------
# Local fine-tuning (one node, one FL round)
# ----------------------------------------------------------------

def local_finetune(model, local_train_df, n_items, device, node_id):
    """
    Fine-tunes a node model for LOCAL_EPOCHS epochs.
    Skipped entirely if config.LOCAL_EPOCHS == 0.
    """
    if config.LOCAL_EPOCHS == 0:
        return model

    _, src, dst = build_local_coo(
        local_train_df,
        local_train_df["user_id"].unique().tolist(),
        model.n_users,
        n_items,
    )

    # Update edge buffers to this node's local graph
    src_t = torch.from_numpy(src.astype(np.int64)).to(device)
    dst_t = torch.from_numpy(dst.astype(np.int64)).to(device)
    n     = model.n_nodes
    deg   = torch.zeros(n, dtype=torch.float32, device=device)
    deg.index_add_(0, src_t, torch.ones(len(src_t), device=device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    ew = deg_inv_sqrt[src_t] * deg_inv_sqrt[dst_t]
    model.edge_src     = src_t
    model.edge_dst     = dst_t
    model.edge_weights = ew

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    model.train()

    for _ in range(config.LOCAL_EPOCHS):
        optimizer.zero_grad()
        u_emb, i_emb = model()
        users_np, pos_np, neg_np = sample_bpr_batch(
            local_train_df, n_items, config.BATCH_SIZE
        )
        users_t = torch.tensor(users_np, dtype=torch.long).to(device)
        pos_t   = torch.tensor(pos_np,   dtype=torch.long).to(device)
        neg_t   = torch.tensor(neg_np,   dtype=torch.long).to(device)
        loss = bpr_loss(u_emb, i_emb, users_t, pos_t, neg_t, config.L2_REG)
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ----------------------------------------------------------------
# Broadcast global weights to all node models
# ----------------------------------------------------------------

def broadcast(node_models, global_state_dict, device):
    """
    Loads global model weights into every node model.
    Non-float buffers (edge_src, edge_dst, edge_weights) are
    preserved from each node's local graph.
    """
    edge_keys = {"edge_src", "edge_dst", "edge_weights"}
    for nid, model in node_models.items():
        local_sd = model.state_dict()
        merged   = {}
        for key, tensor in global_state_dict.items():
            if key in edge_keys:
                # Always keep node-local graph buffers
                merged[key] = local_sd[key]
            else:
                merged[key] = tensor.to(device)
        # Pre-assign edge buffers so shapes match before load_state_dict
        for k in edge_keys:
            if k in merged:
                setattr(model, k, merged[k])
        model.load_state_dict(merged, strict=False)
        model.eval()


# ----------------------------------------------------------------
# Evaluation across all nodes
# ----------------------------------------------------------------

def evaluate_all_nodes(node_models, node_user_map, train_df, test_df, device):
    """
    Evaluates each node on its own users' test interactions.
    Returns mean Recall@K and NDCG@K across all nodes.
    """
    recalls, ndcgs = [], []

    for nid, model in node_models.items():
        user_ids = [int(u) for u in node_user_map.get(str(nid), [])]
        if not user_ids:
            continue
        recall, ndcg = evaluate(
            model, train_df, test_df, user_ids, k=config.TOP_K
        )
        recalls.append(recall)
        ndcgs.append(ndcg)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_ndcg   = float(np.mean(ndcgs))   if ndcgs   else 0.0
    return mean_recall, mean_ndcg


# ----------------------------------------------------------------
# Main FL loop
# ----------------------------------------------------------------

def main():
    t_start = time.time()

    log("=" * 60)
    log("Module 4 — H-FedAvg Federated Learning")
    log("=" * 60)

    device = torch.device("cpu")
    log(f"Device        : {device}")
    log(f"FL rounds     : {config.FL_ROUNDS}")
    log(f"Local epochs  : {config.LOCAL_EPOCHS}")
    log(f"Eval every    : {config.EVAL_EVERY} rounds")

    # ---- Load inputs ----
    stats, cluster_node_map, node_user_map, train_df, test_df, full_src, full_dst = load_inputs()
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    log(f"Users: {n_users}  Items: {n_items}")
    log(f"Clusters: {len(cluster_node_map)}  →  {cluster_node_map}")

    # ---- Load node models ----
    all_node_ids = sorted([int(k) for k in node_user_map.keys()])
    node_models, node_meta = load_node_models(all_node_ids, n_users, n_items, device)

    if not node_models:
        raise RuntimeError("No node models found. Run Module 3 first.")

    # Interaction counts per node (for weighting)
    node_counts = {
        nid: meta.get("local_train_interactions", 1)
        for nid, meta in node_meta.items()
    }

    # Per-node local training dataframes (used in fine-tuning)
    node_train_dfs = {
        nid: train_df[train_df["user_id"].isin(
            [int(u) for u in node_user_map.get(str(nid), [])]
        )].reset_index(drop=True)
        for nid in node_models
    }

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    round_metrics = []

    # ---- Initial evaluation (round 0) ----
    log("\nEvaluating initial models (round 0)...")
    r0_recall, r0_ndcg = evaluate_all_nodes(
        node_models, node_user_map, train_df, test_df, device
    )
    log(f"  Round 0 | Recall@{config.TOP_K}: {r0_recall:.4f} | NDCG@{config.TOP_K}: {r0_ndcg:.4f}")
    round_metrics.append({
        "round":    0,
        "recall":   r0_recall,
        "ndcg":     r0_ndcg,
        "timestamp": _now_iso(),
    })

    # ---- FL rounds ----
    for rnd in range(1, config.FL_ROUNDS + 1):
        t_rnd = time.time()
        log(f"\n--- Round {rnd}/{config.FL_ROUNDS} ---")

        # Step 1: Local fine-tuning
        if config.LOCAL_EPOCHS > 0:
            log(f"  Step 1: local fine-tune ({config.LOCAL_EPOCHS} epochs per node)")
            for nid, model in node_models.items():
                local_finetune(
                    model, node_train_dfs[nid], n_items, device, nid
                )
        else:
            log("  Step 1: local fine-tune skipped (LOCAL_EPOCHS=0)")

        # Step 2: Tier-1 aggregation
        log("  Step 2: Tier-1 aggregation (within clusters)")
        node_state_dicts = {
            nid: model.state_dict() for nid, model in node_models.items()
        }
        cluster_models = tier1_aggregate(
            cluster_node_map, node_state_dicts, node_counts
        )
        log(f"    Produced {len(cluster_models)} cluster models")

        # Step 3: Tier-2 aggregation
        log("  Step 3: Tier-2 aggregation (global model)")
        cluster_counts = {}
        for cid_str, node_ids in cluster_node_map.items():
            cid = int(cid_str)
            cluster_counts[cid] = sum(
                node_counts.get(nid, 0) for nid in node_ids
                if nid in node_models
            )
        global_sd = tier2_aggregate(cluster_models, cluster_counts)
        log(f"    Global model produced")

        # Step 4: Broadcast
        log("  Step 4: Broadcasting global model to all nodes")
        broadcast(node_models, global_sd, device)

        # Step 5: Evaluate
        if rnd % config.EVAL_EVERY == 0 or rnd == config.FL_ROUNDS:
            log(f"  Step 5: Evaluating...")
            recall, ndcg = evaluate_all_nodes(
                node_models, node_user_map, train_df, test_df, device
            )
            log(f"    Round {rnd} | Recall@{config.TOP_K}: {recall:.4f} | NDCG@{config.TOP_K}: {ndcg:.4f}")
            round_metrics.append({
                "round":    rnd,
                "recall":   recall,
                "ndcg":     ndcg,
                "timestamp": _now_iso(),
            })
        else:
            log(f"  Step 5: Eval skipped (next eval at round {(rnd // config.EVAL_EVERY + 1) * config.EVAL_EVERY})")

        log(f"  Round {rnd} done in {time.time() - t_rnd:.1f}s")

    # ---- Save global model ----
    global_model_path = os.path.join(config.MODELS_DIR, "global_lightgcn.pt")
    # Rebuild a full model with node 0's graph buffers as representative
    first_nid   = sorted(node_models.keys())[0]
    global_model_obj = node_models[first_nid]
    torch.save(global_model_obj.state_dict(), global_model_path)
    log(f"\nGlobal model saved → {global_model_path}")

    # ---- Save round metrics ----
    metrics_path = os.path.join(config.MODELS_DIR, "fl_round_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(round_metrics, f, indent=2)
    log(f"Round metrics  → {metrics_path}")

    # ---- Save summary ----
    final = round_metrics[-1]
    summary = {
        "fl_rounds":        config.FL_ROUNDS,
        "local_epochs":     config.LOCAL_EPOCHS,
        "eval_every":       config.EVAL_EVERY,
        "n_nodes":          len(node_models),
        "n_clusters":       len(cluster_node_map),
        f"final_recall@{config.TOP_K}": final["recall"],
        f"final_ndcg@{config.TOP_K}":   final["ndcg"],
        "initial_recall":   round_metrics[0]["recall"],
        "initial_ndcg":     round_metrics[0]["ndcg"],
        "total_time_sec":   round(time.time() - t_start, 1),
        "completed_at":     _now_iso(),
    }
    summary_path = os.path.join(config.MODELS_DIR, "fl_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 60)
    log("Module 4 complete")
    log(f"  Initial  Recall@{config.TOP_K}: {round_metrics[0]['recall']:.4f}")
    log(f"  Final    Recall@{config.TOP_K}: {final['recall']:.4f}")
    log(f"  Initial  NDCG@{config.TOP_K}:  {round_metrics[0]['ndcg']:.4f}")
    log(f"  Final    NDCG@{config.TOP_K}:  {final['ndcg']:.4f}")
    log(f"  Total time : {time.time() - t_start:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
