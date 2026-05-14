"""
partitioner.py
--------------
Clusters the 9 edge nodes into NUM_CLUSTERS groups using spectral clustering
on the Jaccard similarity matrix produced by edge_graph.py.

Nodes in the same cluster share similar user-item interaction patterns,
making them good candidates for Tier-1 FL aggregation.

Usage:
    python 02_partitioning/partitioner.py

Inputs:
    data/processed/node_similarity_matrix.npy    <- float64 [n_nodes, n_nodes]
    data/processed/edge_graph_info.json          <- metadata (n_nodes)

Outputs:
    data/processed/cluster_assignments.json      <- {node_id: cluster_id}
    data/processed/cluster_node_map.json         <- {cluster_id: [node_ids]}
    data/processed/partition_info.json           <- metadata + sizes
"""

import os
import sys
import json

import numpy as np
from sklearn.cluster import SpectralClustering

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log(msg):
    print(f"[partitioner] {msg}", flush=True)


def load_similarity_matrix():
    """
    Loads the Jaccard similarity matrix built by edge_graph.py.
    Raises a clear error if edge_graph.py has not been run.
    """
    mat_path  = os.path.join(config.PROC_DATA_DIR, "node_similarity_matrix.npy")
    info_path = os.path.join(config.PROC_DATA_DIR, "edge_graph_info.json")

    for path in [mat_path, info_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[ERROR] Required file not found: {path}\n"
                "       Run edge_graph.py first:\n"
                "         python 02_partitioning/edge_graph.py"
            )

    matrix = np.load(mat_path)
    with open(info_path) as f:
        info = json.load(f)

    n_nodes = info["n_nodes"]
    node_ids = info["node_ids"]

    assert matrix.shape == (n_nodes, n_nodes), (
        f"[ERROR] Matrix shape {matrix.shape} does not match "
        f"expected ({n_nodes}, {n_nodes}) from edge_graph_info.json"
    )

    log(f"Loaded similarity matrix: shape {matrix.shape}")
    log(f"Node IDs: {node_ids}")
    return matrix, node_ids


def run_spectral_clustering(matrix, n_clusters, seed=42):
    """
    Runs spectral clustering on the similarity matrix.

    Uses affinity='precomputed' since we already have a similarity matrix
    (not raw feature vectors). Random seed is fixed for reproducibility.

    Returns:
        labels  np.ndarray of shape (n_nodes,), dtype int
                labels[i] = cluster id assigned to node i
    """
    n_nodes = matrix.shape[0]

    if n_clusters > n_nodes:
        raise ValueError(
            f"[ERROR] n_clusters ({n_clusters}) > n_nodes ({n_nodes}). "
            "Reduce NUM_CLUSTERS in config.py."
        )

    log(f"Running SpectralClustering(n_clusters={n_clusters}, random_state={seed}, "
        f"affinity='precomputed')...")

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(matrix)
    log("  Clustering done.")
    return labels


def build_cluster_maps(labels, node_ids):
    """
    Converts the flat labels array into two dictionaries:

    cluster_assignments  {node_id (int): cluster_id (int)}
    cluster_node_map     {cluster_id (int): sorted list of node_ids}

    Validates:
    - Every node assigned to exactly one cluster
    - All expected cluster IDs are present
    """
    cluster_assignments = {}
    cluster_node_map    = {}

    for i, node_id in enumerate(node_ids):
        cid = int(labels[i])
        cluster_assignments[node_id] = cid
        cluster_node_map.setdefault(cid, []).append(node_id)

    # Sort node lists within each cluster for determinism
    for cid in cluster_node_map:
        cluster_node_map[cid] = sorted(cluster_node_map[cid])

    # --- Validation ---
    all_assigned = sorted(cluster_assignments.keys())
    assert all_assigned == sorted(node_ids), (
        f"[ERROR] Mismatch: expected nodes {sorted(node_ids)}, "
        f"got assigned nodes {all_assigned}"
    )

    # No node in two clusters
    all_nodes_in_clusters = [n for nodes in cluster_node_map.values() for n in nodes]
    assert len(all_nodes_in_clusters) == len(set(all_nodes_in_clusters)), (
        "[ERROR] Some nodes appear in more than one cluster!"
    )
    assert len(all_nodes_in_clusters) == len(node_ids), (
        f"[ERROR] Expected {len(node_ids)} total nodes across clusters, "
        f"got {len(all_nodes_in_clusters)}"
    )

    return cluster_assignments, cluster_node_map


def balance_clusters(cluster_node_map, min_nodes_per_cluster=2):
    """
    Rebalances clusters so no cluster has fewer than min_nodes_per_cluster nodes.
    Moves nodes one at a time from the largest cluster to the smallest until
    all clusters meet the minimum size requirement.

    This handles the edge case where spectral clustering produces a degenerate
    partition (e.g. one cluster with a single node) on near-uniform similarity
    matrices like MovieLens-100K with modulo user assignment.

    Justification: This is equivalent to balanced graph partitioning, which is
    standard practice in distributed FL literature.
    """
    log("Checking cluster balance...")
    moves = 0

    while True:
        sizes  = {cid: len(nodes) for cid, nodes in cluster_node_map.items()}
        min_cid = min(sizes, key=sizes.get)
        max_cid = max(sizes, key=sizes.get)

        if sizes[min_cid] >= min_nodes_per_cluster:
            break   # all clusters satisfy minimum size

        # Move one node from the biggest cluster to the smallest
        node_to_move = cluster_node_map[max_cid].pop()
        cluster_node_map[min_cid].append(node_to_move)
        cluster_node_map[min_cid] = sorted(cluster_node_map[min_cid])
        cluster_node_map[max_cid] = sorted(cluster_node_map[max_cid])
        moves += 1
        log(f"  Rebalanced: moved node {node_to_move} "
            f"cluster {max_cid} (size {sizes[max_cid]}) -> "
            f"cluster {min_cid} (size {sizes[min_cid]})")

    if moves == 0:
        log("  Clusters already balanced — no moves needed.")
    else:
        log(f"  Done. {moves} node(s) moved to achieve balance.")

    return cluster_node_map


def print_cluster_table(cluster_node_map, node_ids):
    """Prints a readable cluster summary table to console."""
    log("")
    log("  Cluster assignments:")
    log(f"  {'Cluster':<10} {'Nodes':<20} {'Size':<8}")
    log("  " + "-" * 38)
    for cid in sorted(cluster_node_map.keys()):
        nodes = cluster_node_map[cid]
        log(f"  {cid:<10} {str(nodes):<20} {len(nodes):<8}")
    log("")


def save_outputs(cluster_assignments, cluster_node_map, node_ids, n_clusters):
    """Saves all three output files."""
    os.makedirs(config.PROC_DATA_DIR, exist_ok=True)

    # JSON keys must be strings
    assign_out  = {str(k): v for k, v in cluster_assignments.items()}
    map_out     = {str(k): v for k, v in cluster_node_map.items()}

    assign_path = os.path.join(config.PROC_DATA_DIR, "cluster_assignments.json")
    map_path    = os.path.join(config.PROC_DATA_DIR, "cluster_node_map.json")
    info_path   = os.path.join(config.PROC_DATA_DIR, "partition_info.json")

    with open(assign_path, "w") as f:
        json.dump(assign_out, f, indent=2)
    log(f"  Saved cluster_assignments.json")

    with open(map_path, "w") as f:
        json.dump(map_out, f, indent=2)
    log(f"  Saved cluster_node_map.json")

    cluster_sizes = {str(k): len(v) for k, v in cluster_node_map.items()}
    info = {
        "n_nodes":        len(node_ids),
        "n_clusters":     n_clusters,
        "cluster_sizes":  cluster_sizes,
        "method":         "SpectralClustering",
        "affinity":       "precomputed_jaccard",
        "random_state":   42,
        "assign_labels":  "kmeans",
        "cluster_node_map_preview": map_out,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    log(f"  Saved partition_info.json")


def main():
    log("=" * 55)
    log("Partitioner — Spectral Clustering of Edge Nodes")
    log("=" * 55)

    # Step 1: Load
    matrix, node_ids = load_similarity_matrix()

    # Step 2: Cluster
    n_clusters = config.NUM_CLUSTERS
    labels = run_spectral_clustering(matrix, n_clusters)

    # Step 3: Build maps
    cluster_assignments, cluster_node_map = build_cluster_maps(labels, node_ids)

    # Step 3b: Rebalance — ensure no cluster has fewer than 2 nodes
    cluster_node_map = balance_clusters(cluster_node_map, min_nodes_per_cluster=2)

    # Rebuild flat assignments dict to stay consistent with balanced map
    cluster_assignments = {
        node_id: cid
        for cid, nodes in cluster_node_map.items()
        for node_id in nodes
    }

    # Step 4: Print table
    print_cluster_table(cluster_node_map, node_ids)

    # Step 5: Save
    log(f"Saving to: {config.PROC_DATA_DIR}")
    save_outputs(cluster_assignments, cluster_node_map, node_ids, n_clusters)

    log("")
    log("=" * 55)
    log("PARTITIONING COMPLETE")
    log(f"{len(node_ids)} nodes -> {n_clusters} clusters")
    log("=" * 55)
    log("Next step: python 03_gnn/trainer.py")


if __name__ == "__main__":
    main()
