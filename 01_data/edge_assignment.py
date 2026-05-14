"""
edge_assignment.py
------------------
Assigns each user to one of NUM_EDGE_NODES simulated edge nodes.

Assignment strategy: user_id % NUM_EDGE_NODES
  - Simple, deterministic, reproducible
  - Produces balanced node populations (each node gets ~equal users)
  - No geographic simulation needed — modulo is standard in FL simulation papers

Usage:
    python 01_data/edge_assignment.py

Inputs:
    data/processed/stats.json
    data/processed/train.csv      <- to confirm all user_ids

Outputs:
    data/processed/user_edge_map.json   <- {user_id (str): edge_node_id}
    data/processed/node_user_map.json   <- {edge_node_id (str): [user_id, ...]}
    data/processed/edge_stats.json      <- per-node user/interaction counts
"""

# !! MUST be first — fixes macOS OpenMP crash before any library loads.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log(msg):
    print(f"[edge_assignment] {msg}", flush=True)


def load_inputs():
    """
    Loads train.csv to get the full list of user_ids.
    Raises clear errors if preprocess.py hasn't been run.
    """
    train_path = os.path.join(config.PROC_DATA_DIR, "train.csv")
    stats_path = os.path.join(config.PROC_DATA_DIR, "stats.json")

    for path in [train_path, stats_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[ERROR] Required file not found: {path}\n"
                "       Run 'python 01_data/preprocess.py' first."
            )

    train_df = pd.read_csv(train_path)
    with open(stats_path) as f:
        stats = json.load(f)

    log(f"Loaded train.csv: {len(train_df):,} interactions, "
        f"{train_df['user_id'].nunique()} unique users")
    return train_df, stats


def assign_users_to_nodes(train_df, n_nodes):
    """
    Assigns each unique user_id to a node via: node_id = user_id % n_nodes.

    Returns:
        user_edge_map  dict {user_id: edge_node_id}
        node_user_map  dict {edge_node_id: [user_id, ...]}
    """
    log(f"Assigning users to {n_nodes} edge nodes using: node_id = user_id % {n_nodes}")

    unique_users  = sorted(train_df["user_id"].unique())
    user_edge_map = {}
    node_user_map = defaultdict(list)

    for user_id in unique_users:
        node_id = int(user_id) % n_nodes
        user_edge_map[user_id] = node_id
        node_user_map[node_id].append(int(user_id))

    # Verify all nodes got at least one user
    empty_nodes = [n for n in range(n_nodes) if n not in node_user_map]
    if empty_nodes:
        log(f"  WARNING: Nodes with no users assigned: {empty_nodes}")
        log(f"  This can happen if n_users < n_nodes. Consider reducing NUM_EDGE_NODES.")

    log(f"  Total users assigned: {len(user_edge_map)}")
    return user_edge_map, dict(node_user_map)


def compute_edge_stats(train_df, node_user_map):
    """
    Computes per-node statistics:
      - how many users each node has
      - how many train interactions each node has
    Helps verify the assignment is balanced.
    """
    log("Computing per-node statistics...")
    stats_list = []

    for node_id in sorted(node_user_map.keys()):
        users = node_user_map[node_id]
        node_interactions = train_df[train_df["user_id"].isin(users)]
        stats_list.append({
            "node_id":          node_id,
            "n_users":          len(users),
            "n_interactions":   int(len(node_interactions)),
            "n_items":          int(node_interactions["item_id"].nunique()),
            "avg_per_user":     round(len(node_interactions) / len(users), 2),
        })

    return stats_list


def validate_assignment(user_edge_map, node_user_map, stats_list, n_nodes):
    """
    Checks:
    1. All users are assigned exactly once
    2. No user appears in multiple nodes
    3. Population is reasonably balanced (max/min ratio < 2x)
    """
    log("Validating assignment...")

    # Check 1: total user count matches
    total_assigned = sum(len(v) for v in node_user_map.values())
    assert total_assigned == len(user_edge_map), \
        f"[ERROR] User count mismatch: {total_assigned} vs {len(user_edge_map)}"

    # Check 2: no user in two nodes
    all_assigned = [u for users in node_user_map.values() for u in users]
    assert len(all_assigned) == len(set(all_assigned)), \
        "[ERROR] Some users appear in multiple nodes — assignment bug."

    # Check 3: balance check
    counts = [s["n_users"] for s in stats_list]
    if min(counts) > 0:
        balance_ratio = max(counts) / min(counts)
        if balance_ratio > 2.0:
            log(f"  WARNING: Node population imbalance ratio = {balance_ratio:.2f}x "
                f"(max={max(counts)}, min={min(counts)}). "
                f"Consider adjusting NUM_EDGE_NODES.")
        else:
            log(f"  Balance ratio: {balance_ratio:.2f}x (max={max(counts)}, "
                f"min={min(counts)}). Acceptable.")

    log("  Assignment validation passed.")


def print_per_node_table(stats_list):
    """Prints a clean per-node summary table to console."""
    log("")
    log("  Per-Node Assignment Summary:")
    log(f"  {'Node':>5}  {'Users':>7}  {'Interactions':>14}  {'Items':>7}  {'Avg/User':>9}")
    log("  " + "-" * 50)
    for s in stats_list:
        log(f"  {s['node_id']:>5}  {s['n_users']:>7}  "
            f"{s['n_interactions']:>14,}  {s['n_items']:>7}  {s['avg_per_user']:>9}")
    log("")


def save_outputs(user_edge_map, node_user_map, stats_list):
    """Saves all three output files."""
    os.makedirs(config.PROC_DATA_DIR, exist_ok=True)

    uem_path  = os.path.join(config.PROC_DATA_DIR, "user_edge_map.json")
    num_path  = os.path.join(config.PROC_DATA_DIR, "node_user_map.json")
    est_path  = os.path.join(config.PROC_DATA_DIR, "edge_stats.json")

    with open(uem_path, "w") as f:
        json.dump({str(k): v for k, v in user_edge_map.items()}, f)
    with open(num_path, "w") as f:
        # Convert int keys to strings for JSON compatibility
        json.dump({str(k): v for k, v in node_user_map.items()}, f)
    with open(est_path, "w") as f:
        json.dump(stats_list, f, indent=2)

    log(f"  Saved user_edge_map.json  ({os.path.getsize(uem_path) / 1024:.1f} KB)")
    log(f"  Saved node_user_map.json  ({os.path.getsize(num_path) / 1024:.1f} KB)")
    log(f"  Saved edge_stats.json     ({os.path.getsize(est_path) / 1024:.1f} KB)")


def main():
    log("=" * 55)
    log("Edge Assignment — Users to Simulated Edge Nodes")
    log("=" * 55)

    n_nodes = config.NUM_EDGE_NODES
    log(f"NUM_EDGE_NODES = {n_nodes}")

    # Step 1: Load
    train_df, stats = load_inputs()

    # Step 2: Assign
    user_edge_map, node_user_map = assign_users_to_nodes(train_df, n_nodes)

    # Step 3: Compute per-node stats
    stats_list = compute_edge_stats(train_df, node_user_map)

    # Step 4: Validate
    validate_assignment(user_edge_map, node_user_map, stats_list, n_nodes)

    # Step 5: Print table
    print_per_node_table(stats_list)

    # Step 6: Save
    log(f"Saving outputs to: {config.PROC_DATA_DIR}")
    save_outputs(user_edge_map, node_user_map, stats_list)

    log("=" * 55)
    log("EDGE ASSIGNMENT COMPLETE")
    log("=" * 55)
    log("Module 1 done. Next step:")
    log("  python 02_partitioning/edge_graph.py")


if __name__ == "__main__":
    main()
