"""
complexity.py
=============
Theoretical complexity analysis for the Zeroed Out key recovery attacks.
Reproduces all formulae from Section 4.1.2 and 4.2.3 of the paper.

All functions return log2 of the expected number of oracle queries.
"""

from __future__ import annotations
import math
from typing import Tuple


# ---------------------------------------------------------------------------
# Standard wPRF complexity
# ---------------------------------------------------------------------------

def standard_collision_cost(lam: int, c: int, h1: float) -> float:
    """
    Expected oracle queries to accumulate c collisions.
    Uses Generalised Birthday Paradox (Lemma 2):
        S ≈ √(2^{h1+1} · c)

    Parameters
    ----------
    lam : security parameter λ (unused here; kept for API consistency)
    c   : number of collisions desired
    h1  : Hamming weight of key k (h1 ≈ λ for a random key)

    Returns log2 of the expected sample count.
    """
    return (h1 + 1) / 2 + math.log2(c) / 2


def standard_exhaustive_cost(lam: int, c: int, h1: float) -> float:
    """
    Expected oracle queries for exhaustive search after c collisions.
    h1 after c collisions is estimated as h1 + λ/2^c (Equation 1 from paper).
    Search radius d_c = ⌈λ/2^c⌉.

    Returns log2 of expected candidate count ≈ Σ_{j=1}^{d_c} C(H1, j).
    """
    d_c = math.ceil(lam / (2 ** c))
    H1_after_c = h1 + lam / (2 ** c)   # ≈ λ + λ/2^c from Eq.1

    total = sum(math.comb(round(H1_after_c), j) for j in range(1, d_c + 1))
    return math.log2(max(total, 1))


def standard_total_complexity(lam: int) -> Tuple[float, int]:
    """
    Compute the total attack complexity for the Standard One-to-One wPRF.

    Optimises over the transition point C (number of collisions before
    switching to exhaustive search).  From the paper:  C = log2(λ).

    Returns
    -------
    (log2_complexity, optimal_C)
    """
    h1 = float(lam)   # expected Hamming weight for a uniform key
    best_log2 = float('inf')
    best_C = 1

    for C in range(1, int(math.log2(lam)) + 3):
        col_cost = 2 ** standard_collision_cost(lam, C, h1)
        exh_cost = 2 ** standard_exhaustive_cost(lam, C, h1)
        total = col_cost + exh_cost
        log2_total = math.log2(total)
        if log2_total < best_log2:
            best_log2 = log2_total
            best_C = C

    return best_log2, best_C


# ---------------------------------------------------------------------------
# Reversed moduli wPRF complexity
# ---------------------------------------------------------------------------

def reversed_zero_identification_cost(lam: int) -> float:
    """
    Expected oracle queries to find all zero positions in the reversed wPRF.

    im(F) size ≈ 2^{(4λ+3)/6}  (from Section 4.2.1).
    C = O(log3(λ)) collisions needed.
    Total: √(2^{(4λ+3)/3} · C)  = O(2^{2λ/3} · log3(λ))

    Returns log2 of expected query count.
    """
    log23 = math.log2(3)
    # Expected h0^* ≈ 2λ / (3 log2(3))
    h0_star = 2 * lam / (3 * log23)
    C = math.ceil(3 * math.log(h0_star + 1) / math.log(1.5))  # with safety margin
    im_size_bits = (4 * lam + 3) / 3
    return im_size_bits / 2 + math.log2(C) / 2


def reversed_exhaustive_cost(lam: int) -> float:
    """
    Exhaustive search over non-zero key positions.
    |J1 ∪ J2| ≈ (2/3) · n = (2/3) · (2λ / log2(3))
    2^{(2/3)·n} = 2^{(4λ)/(3·log2(3))} ≈ 2^{0.84λ}

    Returns log2 of expected candidate count.
    """
    log23 = math.log2(3)
    n = 2 * lam / log23
    free_positions = (2 / 3) * n
    return free_positions   # log2 of 2^free_positions


def reversed_total_complexity(lam: int) -> float:
    """
    Total attack complexity for the Reversed One-to-One wPRF.
    Dominated by the exhaustive search: O(2^{0.84λ}).
    """
    col_log2 = reversed_zero_identification_cost(lam)
    exh_log2 = reversed_exhaustive_cost(lam)
    total = 2 ** col_log2 + 2 ** exh_log2
    return math.log2(total)


# ---------------------------------------------------------------------------
# Complexity table (prints a LaTeX-ready table)
# ---------------------------------------------------------------------------

def print_complexity_table(lambdas=(16, 20, 24, 28, 32, 34, 40, 64, 80, 128)):
    header = (
        f"{'λ':>5} | "
        f"{'Claimed (λ)':>12} | "
        f"{'Standard attack':>18} | "
        f"{'Reversed attack':>18} | "
        f"{'Opt C (std)':>12}"
    )
    print(header)
    print("-" * len(header))
    for lam in lambdas:
        claimed   = lam
        std_log2, C_opt = standard_total_complexity(lam)
        rev_log2  = reversed_total_complexity(lam)
        print(
            f"{lam:>5} | "
            f"  2^{claimed:<9} | "
            f"  2^{std_log2:<15.2f} | "
            f"  2^{rev_log2:<15.2f} | "
            f"  C={C_opt}"
        )


# ---------------------------------------------------------------------------
# Birthday paradox utilities
# ---------------------------------------------------------------------------

def birthday_samples(output_space_bits: float, num_collisions: int = 1) -> float:
    """
    Expected number of samples to find `num_collisions` collisions
    (Generalised Birthday Paradox, Lemma 2).
    Returns log2 of the sample count.
    """
    return (output_space_bits + 1) / 2 + math.log2(num_collisions) / 2


def hamming_distance_after_collisions(h0: int, c: int) -> float:
    """
    Expected Hamming distance between key estimate K and true key k
    after c collisions.  Equation (1) from the paper:
        d_c = h0 / 2^c
    """
    return h0 / (2 ** c)


if __name__ == "__main__":
    print("=== Complexity Analysis: Zeroed Out Attacks ===\n")
    print_complexity_table()
    print()
    print("=== Birthday Paradox Reference ===")
    for bits in [10, 14, 17, 20, 24]:
        s = birthday_samples(bits)
        print(f"  |Y| = 2^{bits} → first collision after ≈ 2^{s:.2f} samples")
