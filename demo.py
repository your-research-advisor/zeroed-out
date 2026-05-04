#!/usr/bin/env python3
"""
demo.py
=======
Quick demonstration of the Zeroed Out attack.
Suitable for a live presentation or first-run sanity check.

Usage:  python demo.py [--lam 12] [--verbose]
"""

import argparse
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.wprf import WPRF, WPRFParams, WPRFPublicParams
from src.attack import StandardAttack
from src.complexity import standard_total_complexity, print_complexity_table


BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║   Zeroed Out: Cryptanalysis of Weak PRFs in Alternating Moduli  ║
║   Manterola Ayala & Raddum  ·  IACR ToSC 2025                   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def separator(char="─", width=68):
    print(char * width)


def demo_complexity_table():
    print("\n[1] Theoretical attack complexity vs. claimed λ-bit security\n")
    print_complexity_table(lambdas=(8, 10, 12, 16, 20, 24, 28, 32, 34, 40, 64, 128))


def demo_attack(lam: int, verbose: bool):
    print(f"\n[2] Live attack demonstration  (λ = {lam})\n")

    params = WPRFParams.standard_one_to_one(lam)
    pub = WPRFPublicParams(params, seed=2025)
    wprf = WPRF(pub)

    rng = np.random.default_rng(42)
    true_key = wprf.sample_key(rng)

    print(f"  Parameter set    : {params.name}")
    print(f"  Key length n     : {params.n}  (2λ)")
    print(f"  Intermediate dim : {params.m}  (7.06λ)")
    print(f"  Output dim t     : {params.t}  (2λ/log2(3))")
    print(f"  True key k       : {true_key[:8]}... [first 8 bits shown]")
    print(f"  Hamming weight   : {int(np.count_nonzero(true_key))}  "
          f"(zeros: {int(np.sum(true_key == 0))})")

    theoretical_log2, C_opt = standard_total_complexity(lam)
    print(f"\n  Claimed security : 2^{lam} ≈ {2**lam:,}")
    print(f"  Expected attack  : 2^{theoretical_log2:.2f} ≈ {2**theoretical_log2:,.0f}  "
          f"(C_opt = {C_opt} collisions)")
    print(f"\n  Running attack...")

    attack = StandardAttack(wprf, verbose=verbose, rng=np.random.default_rng(99))
    result = attack.run(true_key=true_key)

    separator()
    print(result)
    separator()

    if result.success:
        match = np.array_equal(result.recovered_key, true_key)
        print(f"  Key verified     : {'✓ CORRECT' if match else '✗ WRONG'}")
        speedup = lam - math.log2(max(result.total_queries, 1))
        print(f"  Security loss    : ~2^{speedup:.1f} faster than brute force")
    print()


def demo_structural_insight(lam: int):
    print(f"\n[3] Structural insight: image size shrinks with zero key bits\n")

    params = WPRFParams.standard_one_to_one(lam)
    pub = WPRFPublicParams(params, seed=77)
    wprf = WPRF(pub)
    rng = np.random.default_rng(0)

    n = params.n
    samples = 300

    for zero_fraction, label in [(0.0, "all-ones key"), (0.5, "50% zeros"),
                                  (0.75, "75% zeros"), (0.95, "95% zeros")]:
        k = np.ones(n, dtype=np.int64)
        num_zeros = int(n * zero_fraction)
        k[:num_zeros] = 0

        image = set()
        for _ in range(samples):
            x = wprf.sample_input(rng)
            image.add(wprf.evaluate(k, x).tobytes())

        h1 = int(np.count_nonzero(k))
        print(f"  {label:<20}  h1={h1:>4}  |im(F)|_observed = {len(image):>5}  "
              f"(theoretical 2^{wprf.effective_image_size_bits(k):.1f})")

    print()


def main():
    parser = argparse.ArgumentParser(description="Zeroed Out demo")
    parser.add_argument("--lam", type=int, default=12,
                        help="Security parameter for live attack (default: 12)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-collision details")
    parser.add_argument("--skip-attack", action="store_true",
                        help="Skip the live attack (complexity table + insight only)")
    args = parser.parse_args()

    print(BANNER)
    demo_complexity_table()
    demo_structural_insight(args.lam)
    if not args.skip_attack:
        demo_attack(args.lam, args.verbose)


if __name__ == "__main__":
    main()
