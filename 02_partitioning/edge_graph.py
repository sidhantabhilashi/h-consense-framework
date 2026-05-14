"""
edge_graph.py
-------------
Builds a 9x9 Jaccard similarity matrix between edge nodes.

Two nodes are "similar" if their users share many of the same items.
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
  where A = set of items interacted by users on node A
        B = set of items interacted by users on node B

This matrix is the input to partitioner.py which runs spectral clustering.

Usage:
    python 02_partitioning/edge_graph.py

Inputs:
    data/processed/node_user_map.json    <- {node_id: [user_ids]}
    data/processed/train.csv             <- user_id, item_id

Outputs:
    data/processed/node_similarity_matrix.npy    <- float64 [n_nodes, n_nodes]
    data/processed/node_item_sets.json           <- {node_id: [item_ids]}  (debug)
    data/processed/edge_graph_info.json          <- metadata
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log(msg):
    print(f"[edge_graph] {msg}", flush=True)


def load_inputs():
    """
    Loads node_user_map.json and train.csv.
    Raises a clear error if 01_data hasn't been run.
    """
    num_path   = os.path.join(config.PROC_DATA_DIR, "node_user_map.json")
    train_path = os.path.join(config.PROC_DATA_DIR, "train.csv")

    for path in [num_path, train_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[ERROR] Required file not found: {path}\n"
                "       Run '01_data/' scripts first:\n"
                "         python 01_data/preprocess.py\n"
                "         python 01_data/edge_assignment.py"
            )

    with open(num_path) as f:
        node_user_map = json.load(f)   # keys are strings e.g. "0", "1", ...

    train_df = pd.read_csv(train_path)

    log(f"Loaded node_user_map.json — {len(node_user_map)} nodes")
    log(f"Loaded train.csv — {len(train_df):,} interactions")
    return node_user_map, train_df


def build_node_item_sets(node_user_map, train_df):
    """
    For each edge node, computes the set of items that its users interacted with.

    node_item_sets[node_id] = set of item_ids seen by users on that node

    Returns:
        node_item_sets  dict {int node_id: set of int item_ids}
        sorted_node_ids list of int node_ids in sorted order
    """
    log("Building per-node item sets...")

    # Build a lookup: user_id -> set of item_ids  (faster than repeated df filtering)
    user_items = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    node_item_sets = {}
    sorted_node_ids = sorted([int(k) for k in node_user_map.keys()])

    for node_id in sorted_node_ids:
        users = node_user_map[str(node_id)]   # list of user_ids
        # Union of all items seen by any user on this node
        items = set()
        for uid in users:
            items |= user_items.get(uid, set())
        node_item_sets[node_id] = items
        log(f"  Node {node_id}: {len(users):>4} users, {len(items):>5} unique items")

    return node_item_sets, sorted_node_ids


def jaccard(set_a, set_b):
    """
    Computes Jaccard similarity between two sets.
    Returns 1.0 if both sets are empty (identical empty nodes).
    Returns 0.0 if union is empty but only one side is empty (shouldn't happen).
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    return len(set_a & set_b) / len(union)


def build_similarity_matrix(node_item_sets, sorted_node_ids):
    """
    Builds an n x n Jaccard similarity matrix.
    matrix[i][j] = Jaccard(items on node i, items on node j)

    Properties guaranteed:
    - Symmetric: matrix[i][j] == matrix[j][i]
    - Diagonal:  matrix[i][i] == 1.0
    - Range:     all values in [0.0, 1.0]
    """
    n = len(sorted_node_ids)
    log(f"Computing {n}x{n} Jaccard similarity matrix...")

    matrix = np.zeros((n, n), dtype=np.float64)

    for i, node_i in enumerate(sorted_node_ids):
        for j, node_j in enumerate(sorted_node_ids):
            if i == j:
                matrix[i][j] = 1.0          # diagonal is always 1
            elif j > i:                      # compute upper triangle only
                sim = jaccard(node_item_sets[node_i], node_item_sets[node_j])
                matrix[i][j] = sim
                matrix[j][i] = sim          # mirror to lower triangle
                log(f"  Node {node_i} <-> Node {node_j}: {sim:.4f}")

    return matrix


def validate_matrix(matrix, n_nodes):
    """
    Checks:
    1. Shape is (n_nodes, n_nodes)
    2. Symmetric
    3. Diagonal is all 1.0
    4. All values in [0, 1]
    """
    log("Validating similarity matrix...")

    assert matrix.shape == (n_nodes, n_nodes), \
        f"[ERROR] Expected shape ({n_nodes},{n_nodes}), got {matrix.shape}"

    is_symmetric = np.allclose(matrix, matrix.T, atol=1e-10)
    if not is_symmetric:
        log("  WARNING: Matrix is not perfectly symmetric. Max diff: "
            f"{np.max(np.abs(matrix - matrix.T)):.2e}")
    else:
        log("  Symmetric: True")

    diag_all_one = np.allclose(np.diag(matrix), 1.0, atol=1e-10)
    log(f"  Diagonal all 1.0: {diag_all_one}")
    if not diag_all_one:
        log(f"  WARNING: Diagonal values: {np.diag(matrix)}")

    in_range = (matrix.min() >= 0.0) and (matrix.max() <= 1.0)
    log(f"  Values in [0,1]: {in_range}  "
        f"(min={matrix.min():.4f}, max={matrix.max():.4f})")

    log("  Validation complete.")


def save_outputs(matrix, node_item_sets, sorted_node_ids):
    """Saves matrix, item sets (as lists for JSON), and metadata."""
    os.makedirs(config.PROC_DATA_DIR, exist_ok=True)

    mat_path  = os.path.join(config.PROC_DATA_DIR, "node_similarity_matrix.npy")
    sets_path = os.path.join(config.PROC_DATA_DIR, "node_item_sets.json")
    info_path = os.path.join(config.PROC_DATA_DIR, "edge_graph_info.json")

    np.save(mat_path, matrix)
    log(f"  Saved node_similarity_matrix.npy  ({os.path.getsize(mat_path) / 1024:.1f} KB)")

    # Convert sets to sorted lists for JSON serialisation
    sets_serialisable = {str(k): sorted(list(v)) for k, v in node_item_sets.items()}
    with open(sets_path, "w") as f:
        json.dump(sets_serialisable, f)
    log(f"  Saved node_item_sets.json")

    info = {
        "n_nodes":        len(sorted_node_ids),
        "node_ids":       sorted_node_ids,
        "matrix_shape":   list(matrix.shape),
        "matrix_min":     float(matrix.min()),
        "matrix_max":     float(matrix.max()),
        "similarity_metric": "Jaccard",
        "note": "matrix[i][j] = |items_i ∩ items_j| / |items_i ∪ items_j|",
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    log(f"  Saved edge_graph_info.json")


def main():
    log("=" * 55)
    log("Edge Graph — Node Similarity Matrix (Jaccard)")
    log("=" * 55)

    # Step 1: Load
    node_user_map, train_df = load_inputs()

    # Step 2: Build per-node item sets
    node_item_sets, sorted_node_ids = build_node_item_sets(node_user_map, train_df)

    # Step 3: Build similarity matrix
    matrix = build_similarity_matrix(node_item_sets, sorted_node_ids)

    # Step 4: Validate
    validate_matrix(matrix, len(sorted_node_ids))

    # Step 5: Save
    log(f"Saving to: {config.PROC_DATA_DIR}")
    save_outputs(matrix, node_item_sets, sorted_node_ids)

    log("")
    log("=" * 55)
    log("EDGE GRAPH COMPLETE")
    log("=" * 55)
    log("Next step: python 02_partitioning/partitioner.py")


if __name__ == "__main__":
    main()
