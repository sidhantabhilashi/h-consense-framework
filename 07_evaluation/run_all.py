"""
run_all.py
----------
Module 7 orchestrator — runs all 5 evaluation experiments in sequence.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \\
    python3 07_evaluation/run_all.py

Expected runtime: < 3 minutes.

Outputs (all under data/):
    results/e1_cache_hit_rate.csv
    results/e2_ndcg_convergence.csv
    results/e3_comm_cost.csv
    results/e4_scalability.csv
    results/e5_cluster_size.csv
    plots/e1_cache_hit_rate.png
    plots/e2_ndcg_convergence.png
    plots/e3_comm_cost.png
    plots/e4_scalability.png
    plots/e5_cluster_size.png
"""

import os
import sys
import time
import json
from datetime import datetime, timezone

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
import config


def _separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _run_experiment(name: str, fn):
    """Run one experiment, catch errors, return (result, elapsed_sec)."""
    print(f"\n[run_all] Starting {name}...")
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"[run_all] {name} ✓ completed in {elapsed}s")
        return result, elapsed, None
    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"[run_all] {name} ✗ FAILED in {elapsed}s: {e}")
        return None, elapsed, str(e)


def main():
    wall_start = time.perf_counter()

    _separator("Module 7 — Evaluation")
    print(f"[run_all] Started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"[run_all] Results → {config.RESULTS_DIR}")
    print(f"[run_all] Plots   → {os.path.join(PROJECT_ROOT, 'data', 'plots')}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "plots"), exist_ok=True)

    # ── Import experiments ───────────────────────────────────────
    import e1_cache_hit_rate
    import e2_ndcg_convergence
    import e3_comm_cost
    import e4_scalability
    import e5_cluster_size

    experiments = [
        ("E1 — Cache Hit Rate",       e1_cache_hit_rate.run),
        ("E2 — NDCG Convergence",     e2_ndcg_convergence.run),
        ("E3 — Communication Cost",   e3_comm_cost.run),
        ("E4 — Scalability",          e4_scalability.run),
        ("E5 — Cluster Size Effect",  e5_cluster_size.run),
    ]

    summary = {}
    for name, fn in experiments:
        result, elapsed, error = _run_experiment(name, fn)
        summary[name] = {"elapsed_sec": elapsed, "status": "ok" if error is None else "failed", "error": error}

    # ── Final summary table ─────────────────────────────────────
    wall_elapsed = round(time.perf_counter() - wall_start, 1)
    _separator("Module 7 Complete")
    print(f"{'Experiment':<35} {'Status':>8}  {'Time':>6}")
    print("-" * 55)
    for name, info in summary.items():
        status_icon = "✓" if info["status"] == "ok" else "✗ FAILED"
        print(f"  {name:<33} {status_icon:>8}  {info['elapsed_sec']:>5.1f}s")
    print("-" * 55)
    print(f"  Total wall time: {wall_elapsed}s")

    # ── Check output files exist ───────────────────────────────
    print("\n[run_all] Output files:")
    expected = [
        ("results", "e1_cache_hit_rate.csv"),
        ("results", "e2_ndcg_convergence.csv"),
        ("results", "e3_comm_cost.csv"),
        ("results", "e4_scalability.csv"),
        ("results", "e5_cluster_size.csv"),
        ("plots",   "e1_cache_hit_rate.png"),
        ("plots",   "e2_ndcg_convergence.png"),
        ("plots",   "e3_comm_cost.png"),
        ("plots",   "e4_scalability.png"),
        ("plots",   "e5_cluster_size.png"),
    ]
    all_ok = True
    for subdir, fname in expected:
        fpath = os.path.join(PROJECT_ROOT, "data", subdir, fname)
        exists = os.path.exists(fpath)
        icon   = "✓" if exists else "✗ MISSING"
        print(f"  {icon}  data/{subdir}/{fname}")
        if not exists:
            all_ok = False

    # ── Save evaluation summary JSON ───────────────────────────
    eval_summary = {
        "experiments": summary,
        "all_outputs_present": all_ok,
        "total_time_sec": wall_elapsed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = os.path.join(config.RESULTS_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\n[run_all] Summary JSON → {summary_path}")

    if all_ok and all(v["status"] == "ok" for v in summary.values()):
        print("\n[run_all] ✅ Module 7 COMPLETE — all experiments passed.")
    else:
        print("\n[run_all] ⚠  Module 7 finished with errors — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
