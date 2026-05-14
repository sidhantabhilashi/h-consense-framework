# H-GNN-Consense Framework

**Hierarchical GNN-Based Intelligent Edge Caching**
SIT724 Honours Thesis — Sidhant Abhilashi (s224244319)
Supervised by: Dr. Feifei Chen, Deakin University

---

## Project Overview

H-GNN-Consense integrates four components to achieve intelligent, privacy-preserving edge caching:

1. **LightGCN Demand Prediction** — predicts content popularity per edge node
2. **Graph Partitioning** — clusters edge nodes to minimise inter-cluster traffic
3. **Hierarchical Federated Learning** — two-tier H-FedAvg aggregation
4. **Demand-Aware Dissemination** — proactive content push based on predicted demand

---

## Folder Structure

```
thesis/code/
├── config.py                  ← All hyperparameters & paths (edit here)
├── requirements.txt
├── README.md
├── 01_data/                   ← Dataset loading, graph building, edge assignment
├── 02_partitioning/           ← Spectral clustering of edge nodes into clusters
├── 03_gnn/                    ← LightGCN model + BPR training
├── 04_federated/              ← Hierarchical FL simulation (Tier-1 + Tier-2)
├── 05_dissemination/          ← Cache simulator + demand-aware push protocol
├── 06_baselines/              ← LRU, LFU, Flat FedAvg comparison baselines
├── 07_evaluation/             ← Experiments + plot generation
└── results/
    ├── logs/                  ← JSON experiment logs
    └── plots/                 ← Output PNG figures
```

---

## Quick Start

### 1. Install dependencies
```bash
cd thesis/code
pip install -r requirements.txt
```

### 2. Download & preprocess data
```bash
python 01_data/download_movielens.py
python 01_data/preprocess.py
python 01_data/graph_builder.py
python 01_data/edge_assignment.py
```

### 3. Run partitioning
```bash
python 02_partitioning/edge_graph.py
python 02_partitioning/partitioner.py
```

### 4. Run full experiment (all 5 evaluations)
```bash
python 07_evaluation/run_experiment.py --all
```

### 5. Generate plots
```bash
python 07_evaluation/plot_results.py
```
Plots saved to `results/plots/`

---

## Configuration

All settings are in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_EDGE_NODES` | 9 | Simulated edge nodes |
| `NUM_CLUSTERS` | 3 | K for partitioning |
| `CACHE_CAPACITY` | 50 | Items per node cache |
| `FL_ROUNDS` | 50 | Global FL rounds |
| `EMBED_DIM` | 64 | LightGCN embedding size |
| `LOCAL_EPOCHS` | 5 | Training epochs per FL round |
| `DEMAND_THRESHOLD` | 0.5 | θ for dissemination push |
| `SEED` | 42 | Random seed (reproducibility) |

---

## Experiments

| ID | Name | Variable | Metric |
|----|------|----------|--------|
| E1 | Hit Rate vs Method | Algorithm | Cache Hit Rate |
| E2 | Convergence | FL Rounds | NDCG@10 |
| E3 | Comm. Cost | Algorithm | Comm. Rounds |
| E4 | Scalability | N nodes | Hit Rate |
| E5 | Cluster Size | K | Hit Rate |

---

## Research Questions

- **RQ1** — How does partitioning affect scalability?
- **RQ2** — Does GNN prediction improve cache hit rate vs LRU/LFU?
- **RQ3** — How does cluster size K affect performance?
- **RQ4** — Does H-FL reduce communication cost vs standard FedAvg?
- **RQ5** — Does demand-aware dissemination reduce latency?

---

## Reproducibility

All experiments use `SEED = 42`. Running `python 07_evaluation/run_experiment.py --all`
three times should produce identical results.
