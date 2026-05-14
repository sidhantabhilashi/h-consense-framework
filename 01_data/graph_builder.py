"""
graph_builder.py
----------------
Builds a user-item bipartite interaction graph from train.csv.

NO PyTorch used here — torch is not needed for data prep.
Saves as .npy files. Module 3 (lightgcn.py) loads them and converts
to tensors when it needs them. This avoids a macOS segfault caused by
large-array torch.tensor() calls with NumPy 2.x on Apple Silicon.

Graph format (COO = Coordinate format):
  - Users occupy node indices   0 .. N_users - 1
  - Items occupy node indices   N_users .. N_users + N_items - 1
  - Each (user, item) interaction = TWO directed edges (user->item + item->user)
    This is standard for LightGCN.

Usage:
    python 01_data/graph_builder.py

Inputs:
    data/processed/train.csv
    data/processed/stats.json

Outputs:
    data/processed/graph_src.npy         <- int64 array [2 * n_interactions]
    data/processed/graph_dst.npy         <- int64 array [2 * n_interactions]
    data/processed/graph_info.json       <- metadata
"""

# Fix macOS OpenMP crash (libomp conflict between Homebrew Python + PyTorch)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log(msg):
    print(f"[graph_builder] {msg}", flush=True)


def load_inputs():
    """
    Loads train.csv and stats.json.
    Raises a clear error if preprocess.py hasn't been run.
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

    log(f"Loaded train.csv:  {len(train_df):,} interactions")
    log(f"Stats — users: {stats['n_users']}, items: {stats['n_items']}")
    return train_df, stats


def build_bipartite_coo(train_df, n_users):
    """
    Builds COO (source, destination) arrays for a bipartite graph.

    Node IDs:
      - User  u  ->  node index  u
      - Item  i  ->  node index  n_users + i

    For each (user_id, item_id) pair we create TWO directed edges:
      - user -> item   (user to item)
      - item -> user   (item to user, symmetric for LightGCN propagation)

    Returns:
        src  np.ndarray int64  [2 * n_interactions]
        dst  np.ndarray int64  [2 * n_interactions]
    """
    log("Building bipartite COO arrays (bidirectional, no torch)")

    # Extract as plain Python lists first — avoids ALL numpy/torch compat issues
    user_ids = train_df["user_id"].tolist()  # list of ints
    item_ids = train_df["item_id"].tolist()  # list of ints

    n = len(user_ids)

    # Pre-allocate output arrays
    src = np.empty(2 * n, dtype=np.int64)
    dst = np.empty(2 * n, dtype=np.int64)

    # Fill user -> item edges (first half)
    for i in range(n):
        src[i] = user_ids[i]
        dst[i] = n_users + item_ids[i]

    # Fill item -> user edges (second half)
    for i in range(n):
        src[n + i] = n_users + item_ids[i]
        dst[n + i] = user_ids[i]

    log(f"  src shape: {src.shape}, dtype: {src.dtype}")
    log(f"  dst shape: {dst.shape}, dtype: {dst.dtype}")
    log(f"  Node count: {n_users} users + {train_df['item_id'].nunique()} items "
        f"= {n_users + train_df['item_id'].nunique()} total nodes")

    return src, dst


def validate_coo(src, dst, n_nodes):
    """
    Sanity checks:
    - src and dst have same length
    - All indices in valid range [0, n_nodes)
    - No self-loops (bipartite graph should not have any)
    """
    log("Validating COO arrays...")

    assert len(src) == len(dst), \
        f"[ERROR] src and dst have different lengths: {len(src)} vs {len(dst)}"

    min_idx = int(min(src.min(), dst.min()))
    max_idx = int(max(src.max(), dst.max()))

    assert min_idx >= 0, \
        f"[ERROR] Negative node index found: {min_idx}"
    assert max_idx < n_nodes, \
        f"[ERROR] Node index {max_idx} out of range (n_nodes={n_nodes})"

    self_loops = int((src == dst).sum())
    if self_loops > 0:
        log(f"  WARNING: {self_loops} self-loops found.")
    else:
        log("  No self-loops found. Good.")

    log(f"  Node index range: [{min_idx}, {max_idx}] (n_nodes={n_nodes}). Valid.")


def save_outputs(src, dst, stats):
    """
    Saves src and dst as .npy files and writes graph_info.json.
    Module 3 (LightGCN) will load these and convert to torch tensors there,
    where torch is expected to be imported in a controlled way.
    """
    os.makedirs(config.PROC_DATA_DIR, exist_ok=True)

    src_path  = os.path.join(config.PROC_DATA_DIR, "graph_src.npy")
    dst_path  = os.path.join(config.PROC_DATA_DIR, "graph_dst.npy")
    info_path = os.path.join(config.PROC_DATA_DIR, "graph_info.json")

    np.save(src_path, src)
    np.save(dst_path, dst)

    log(f"  Saved graph_src.npy  ({os.path.getsize(src_path)  / 1024:.1f} KB)")
    log(f"  Saved graph_dst.npy  ({os.path.getsize(dst_path)  / 1024:.1f} KB)")

    n_users = stats["n_users"]
    n_items = stats["n_items"]
    graph_info = {
        "n_users":           n_users,
        "n_items":           n_items,
        "n_nodes":           n_users + n_items,
        "n_edges":           len(src),
        "n_interactions":    stats["n_train_interactions"],
        "format":            "COO — src.npy + dst.npy, int64, no self-loops",
        "node_convention":   "users: 0..n_users-1 | items: n_users..n_users+n_items-1",
        "note":              "Bidirectional. Load with np.load(). Convert to torch in Module 3.",
    }
    with open(info_path, "w") as f:
        json.dump(graph_info, f, indent=2)
    log(f"  Saved graph_info.json")
    return graph_info


def main():
    log("=" * 55)
    log("Graph Builder — Bipartite User-Item Graph (numpy only)")
    log("=" * 55)
    log("NOTE: No PyTorch used here — tensors are built in Module 3.")

    # Step 1: Load inputs
    train_df, stats = load_inputs()

    n_users = stats["n_users"]
    n_nodes = stats["n_users"] + stats["n_items"]

    # Step 2: Build COO arrays (pure numpy + python lists)
    src, dst = build_bipartite_coo(train_df, n_users)

    # Step 3: Validate
    validate_coo(src, dst, n_nodes)

    # Step 4: Save
    log(f"Saving outputs to: {config.PROC_DATA_DIR}")
    graph_info = save_outputs(src, dst, stats)

    log("")
    log("=" * 55)
    log("GRAPH BUILD COMPLETE — Summary")
    log("=" * 55)
    log(f"  Total nodes:   {graph_info['n_nodes']}  "
        f"({graph_info['n_users']} users + {graph_info['n_items']} items)")
    log(f"  Total edges:   {graph_info['n_edges']:,}  "
        f"(2 x {graph_info['n_interactions']:,} interactions, bidirectional)")
    log("=" * 55)
    log("Next step: python 01_data/edge_assignment.py")


if __name__ == "__main__":
    main()
