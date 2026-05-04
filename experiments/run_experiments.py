"""
experiments/run_experiments.py
================================
Reproduce the experimental results from Section 5 of the paper.

Runs N independent attack trials for each λ and reports:
  - C_col   : average oracle queries in collision phase
  - C_exs   : average oracle queries in exhaustive phase
  - C_tot   : total average queries
  - #Coll   : average number of collisions to recover key
  - Acc(C)  : success rate of transition point C (Inequality 3)

Usage
-----
    python experiments/run_experiments.py [--lam 28 34] [--trials 100] [--verbose]

Outputs:
  - Console table matching Table 2 of the paper
  - figures/hamming_distance.png  (matching Figure 2)
  - results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.wprf import WPRF, WPRFParams, WPRFPublicParams
from src.attack import StandardAttack, AttackResult
from src.complexity import standard_total_complexity


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single(lam: int, seed: int, verbose: bool = False) -> AttackResult:
    params = WPRFParams.standard_one_to_one(lam)
    pub = WPRFPublicParams(params, seed=seed)
    wprf = WPRF(pub)

    rng_key = np.random.default_rng(seed * 1000 + 1)
    rng_atk = np.random.default_rng(seed * 1000 + 2)

    true_key = wprf.sample_key(rng_key)
    attack = StandardAttack(wprf, verbose=verbose, rng=rng_atk)
    return attack.run(true_key=true_key)


# ---------------------------------------------------------------------------
# Multi-trial experiment
# ---------------------------------------------------------------------------

def run_experiment(lam: int, num_trials: int, verbose: bool = False) -> dict:
    """
    Run `num_trials` independent attacks for a given λ.
    Returns a dict of aggregate statistics.
    """
    results: List[AttackResult] = []
    theoretical_log2, C_opt = standard_total_complexity(lam)

    print(f"\n  λ={lam}  ({num_trials} trials, "
          f"theoretical complexity ≈ 2^{theoretical_log2:.2f})")
    print(f"  {'Trial':>6}  {'C_col':>10}  {'C_exs':>10}  {'C_tot':>10}  "
          f"{'#Coll':>6}  {'OK':>4}")
    print("  " + "-" * 56)

    for trial in range(num_trials):
        r = run_single(lam, seed=trial, verbose=False)
        results.append(r)
        if verbose or trial < 5 or trial == num_trials - 1:
            status = "✓" if r.success else "✗"
            print(f"  {trial:>6}  "
                  f"2^{math.log2(max(r.queries_collision_phase,1)):>7.2f}  "
                  f"2^{math.log2(max(r.queries_exhaustive_phase,1)):>7.2f}  "
                  f"2^{math.log2(max(r.total_queries,1)):>7.2f}  "
                  f"{r.num_collisions:>6}  "
                  f"{status:>4}")

    successes = [r for r in results if r.success]
    success_rate = len(successes) / num_trials * 100

    def safe_log2(x): return math.log2(x) if x > 0 else 0.0

    def avg_log2(vals):
        vals = [v for v in vals if v > 0]
        return safe_log2(np.mean(vals)) if vals else 0.0

    c_cols = [r.queries_collision_phase for r in results]
    c_exss = [r.queries_exhaustive_phase for r in results]
    c_tots = [r.total_queries for r in results]
    colls  = [r.num_collisions for r in results]

    # Transition point accuracy: fraction where the computed C (Inequality 3)
    # led to correct key on first exhaustive attempt
    # (We infer this: if num_collisions ≈ C_opt, the transition was accurate)
    transition_successes = sum(
        1 for r in results
        if r.success and abs(r.num_collisions - C_opt) <= 2
    )
    transition_accuracy = transition_successes / num_trials * 100

    # Collect Hamming distance traces (use successes only)
    all_hd_traces = [r.hamming_distances for r in successes if r.hamming_distances]

    agg = {
        "lam": lam,
        "num_trials": num_trials,
        "success_rate_pct": success_rate,
        "C_col_log2_mean": avg_log2(c_cols),
        "C_exs_log2_mean": avg_log2(c_exss),
        "C_tot_log2_mean": avg_log2(c_tots),
        "num_collisions_mean": float(np.mean(colls)),
        "transition_accuracy_pct": transition_accuracy,
        "theoretical_log2": theoretical_log2,
        "C_opt": C_opt,
        "hamming_traces": all_hd_traces,
    }

    print(f"\n  Summary for λ={lam}:")
    print(f"    Success rate         : {success_rate:.1f}%")
    print(f"    C_col  (avg log2)    : {agg['C_col_log2_mean']:.2f}  "
          f"(theoretical: {(lam+1)/2:.2f})")
    print(f"    C_exs  (avg log2)    : {agg['C_exs_log2_mean']:.2f}")
    print(f"    C_tot  (avg log2)    : {agg['C_tot_log2_mean']:.2f}  "
          f"(theoretical: {theoretical_log2:.2f})")
    print(f"    Mean collisions      : {agg['num_collisions_mean']:.2f}")
    print(f"    Transition accuracy  : {transition_accuracy:.1f}%")

    return agg


# ---------------------------------------------------------------------------
# Figure 2: Hamming distance vs. collision step
# ---------------------------------------------------------------------------

def plot_hamming_distance(agg: dict, output_dir: str = "figures"):
    """Plot average Hamming distance vs. collision number (Figure 2)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available — skipping Hamming plot]")
        return

    traces = agg["hamming_traces"]
    if not traces:
        print("  [No Hamming traces collected — skipping plot]")
        return

    # Align traces to same length (pad with last value)
    max_len = max(len(t) for t in traces)
    padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
    avg_trace = np.mean(padded, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(avg_trace) + 1), avg_trace,
            marker="o", color="#1f77b4", linewidth=2, markersize=6)
    ax.set_xlabel("Collision step", fontsize=12)
    ax.set_ylabel("Average Hamming distance", fontsize=12)
    ax.set_title(f"Hamming distance decay, λ={agg['lam']}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"hamming_distance_lam{agg['lam']}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Print paper-style Table 2
# ---------------------------------------------------------------------------

def print_table(aggregates: List[dict]):
    print("\n" + "=" * 78)
    print("Table 2 — Experimental Results (reproduction)")
    print("=" * 78)
    header = (f"{'λ':>4} | "
              f"{'C_col':>10} | "
              f"{'C_exs':>10} | "
              f"{'C_tot':>10} | "
              f"{'Theoretical':>12} | "
              f"{'#Coll':>6} | "
              f"{'Acc(C)%':>8}")
    print(header)
    print("-" * 78)
    for agg in aggregates:
        print(f"{agg['lam']:>4} | "
              f"2^{agg['C_col_log2_mean']:>7.2f}  | "
              f"2^{agg['C_exs_log2_mean']:>7.2f}  | "
              f"2^{agg['C_tot_log2_mean']:>7.2f}  | "
              f"  2^{agg['theoretical_log2']:>7.2f}  | "
              f"{agg['num_collisions_mean']:>6.2f} | "
              f"{agg['transition_accuracy_pct']:>7.1f}%")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Zeroed Out experimental results (Section 5)."
    )
    parser.add_argument(
        "--lam", nargs="+", type=int, default=[28, 34],
        help="Security parameters to test (default: 28 34)"
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of independent trials per λ (default: 100; paper uses 1000)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each trial's result"
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="Directory for results.json and figures/ (default: current dir)"
    )
    args = parser.parse_args()

    print("Zeroed Out: Experimental Verification")
    print("=" * 40)
    print(f"Security parameters : λ ∈ {args.lam}")
    print(f"Trials per λ        : {args.trials}")
    print()

    aggregates = []
    for lam in args.lam:
        agg = run_experiment(lam, args.trials, verbose=args.verbose)
        aggregates.append(agg)
        plot_hamming_distance(agg, output_dir=os.path.join(args.output, "figures"))

    print_table(aggregates)

    # Save JSON
    out_json = os.path.join(args.output, "results.json")
    # Remove numpy arrays before serialising
    clean = []
    for agg in aggregates:
        c = {k: v for k, v in agg.items() if k != "hamming_traces"}
        clean.append(c)
    with open(out_json, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
