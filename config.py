"""
config.py
---------
Single source of truth for ALL project-wide settings.
Every module imports this file instead of hardcoding paths or values.

If you want to change the number of edge nodes, FL rounds, embedding dim, etc.,
change it here and everything else updates automatically.
"""

import os

# ---------------------------------------------------------------
# macOS ARM + PyTorch 2.11 OpenMP fix
# ---------------------------------------------------------------
# Two libomp.dylib are present: PyTorch bundled + Homebrew.
# When OMP spawns worker threads the duplicate mutex init segfaults.
# Fix: disable OMP threading (single-threaded is fine for our
# small models) and allow duplicate libs as a fallback.
# MUST be set before any `import torch`.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ============================================================
# PATHS — all relative to this config.py file
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DATA_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR      = os.path.join(PROJECT_ROOT, "data", "models")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "data", "results")
LOGS_DIR        = os.path.join(PROJECT_ROOT, "logs")

# ============================================================
# DATASET
# ============================================================
MOVIELENS_URL  = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_ZIP  = "ml-100k.zip"
MOVIELENS_DIR  = "ml-100k"
RATINGS_FILE   = "u.data"

# Minimum interactions to keep a user/item after filtering
MIN_USER_RATINGS = 5
MIN_ITEM_RATINGS = 5

# Train/test split ratios (per user)
TRAIN_RATIO = 0.8
TEST_RATIO  = 0.2

# Global random seed for reproducibility
SEED = 42

# ============================================================
# SIMULATED NETWORK TOPOLOGY
# ============================================================
# Number of edge nodes in the simulated network
NUM_EDGE_NODES = 9

# Number of clusters (K in K-means / spectral clustering)
NUM_CLUSTERS = 3

# ============================================================
# GNN MODEL (LightGCN)
# ============================================================
EMBEDDING_DIM  = 64      # size of user/item embedding vectors
NUM_GNN_LAYERS  = 3      # number of LightGCN propagation layers
DROPOUT         = 0.0    # LightGCN uses no dropout by default

# Module 3 shorthand aliases (used directly by 03_gnn/trainer.py)
EMB_DIM    = EMBEDDING_DIM    # 64
N_LAYERS   = NUM_GNN_LAYERS   # 3
N_EPOCHS   = 50               # standalone training epochs (not FL local epochs)
LR         = 1e-3             # Adam learning rate
BATCH_SIZE = 1024             # BPR batch size for standalone training
L2_REG     = 1e-4             # L2 regularisation coefficient
LOG_EVERY  = 10               # print loss every N epochs
# TOP_K already defined in EVALUATION section below
# MODELS_DIR already defined in PATHS section above

# ============================================================
# FEDERATED LEARNING
# ============================================================
FL_ROUNDS        = 20    # number of global FL rounds
LOCAL_EPOCHS     = 5     # local training epochs per round per node
EVAL_EVERY       = 5     # evaluate all nodes every N FL rounds
LEARNING_RATE    = 1e-3  # Adam optimizer learning rate
BATCH_SIZE       = 512   # BPR training batch size
NEG_SAMPLES      = 1     # number of negative samples per positive (BPR)

# ============================================================
# DISSEMINATION / CACHE
# ============================================================
CACHE_SIZE_K     = 50    # top-K items to proactively cache per edge node
DEMAND_THRESHOLD = 0.5   # minimum predicted demand score to trigger push

# ============================================================
# EVALUATION
# ============================================================
TOP_K            = 10    # K for NDCG@K, Hit Rate@K

# ============================================================
# DEBUG / LOGGING
# ============================================================
# Set to True to enable extra verbose prints inside each module
DEBUG = False
