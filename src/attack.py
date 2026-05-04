"""
attack.py
=========
Complete implementation of the key-recovery attack against the APRR24
One-to-One wPRF parameter sets, as described in:

    "Zeroed Out: Cryptanalysis of Weak PRFs in Alternating Moduli"
    Manterola Ayala & Raddum, IACR ToSC 2025, Vol. 2025, No. 2.

Two attack classes are provided:

    StandardAttack   — targets the (F2, F3)-wPRF, One-to-One parameters
                       Complexity: O(2^{λ/2} · log2(λ))

    ReversedAttack   — targets the (F3, F2)-wPRF, One-to-One parameters
                       Complexity: O(2^{0.84λ})

Both follow the same conceptual template:
  Phase 1  Accumulate output collisions to identify zero positions in k.
  Phase 2  Exhaustive search over remaining key candidates.
"""

from __future__ import annotations

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from collections import defaultdict

from .wprf import WPRF, WPRFPublicParams, WPRFParams


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AttackResult:
    """
    Records the full outcome and statistics of one attack run.
    """
    success: bool
    recovered_key: Optional[np.ndarray]
    true_key: Optional[np.ndarray]

    # Oracle query counts
    queries_collision_phase: int = 0
    queries_exhaustive_phase: int = 0

    # Collision trace
    num_collisions: int = 0
    hamming_distances: list = field(default_factory=list)

    # Timing
    elapsed_seconds: float = 0.0

    @property
    def total_queries(self) -> int:
        return self.queries_collision_phase + self.queries_exhaustive_phase

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        lines = [
            f"Attack Result: {status}",
            f"  Total oracle queries : {self.total_queries:>10,}  (≈ 2^{math.log2(max(self.total_queries,1)):.2f})",
            f"  Collision phase      : {self.queries_collision_phase:>10,}  (≈ 2^{math.log2(max(self.queries_collision_phase,1)):.2f})",
            f"  Exhaustive phase     : {self.queries_exhaustive_phase:>10,}  (≈ 2^{math.log2(max(self.queries_exhaustive_phase,1)):.2f})",
            f"  Collisions found     : {self.num_collisions}",
            f"  Elapsed              : {self.elapsed_seconds:.3f}s",
        ]
        if self.hamming_distances:
            hd_str = " → ".join(str(d) for d in self.hamming_distances)
            lines.append(f"  Hamming dist trace   : {hd_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: output tuple → hashable key
# ---------------------------------------------------------------------------

def _to_key(y: np.ndarray) -> bytes:
    return y.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# Standard (F2, F3) One-to-One attack
# ---------------------------------------------------------------------------

class StandardAttack:
    """
    Key recovery attack on the Standard (F2, F3)-wPRF with One-to-One params.

    Core observation
    ----------------
    For a fixed key k ∈ {0,1}^n, any position i with k_i = 0 makes x_i
    irrelevant: (k ⊙ x)_i = 0 regardless of x_i.  This collapses the
    effective input space to 2^{h1} where h1 = HammingWeight(k) ≈ λ.

    By the birthday paradox we find the first collision after ≈ 2^{h1/2}
    queries.  Each collision (x, x') with F(k,x)=F(k,x') reveals that every
    position where x_i ≠ x_i' must have k_i = 0.

    After C = ⌈log2(λ)⌉ collisions the Hamming distance between our running
    key estimate K and the true key k satisfies d_C ≈ λ/2^C = 1.  We then
    switch to exhaustive search over the ~λ remaining 1-bit candidates, which
    costs O(λ) additional queries.

    Total: O(2^{λ/2} · log2(λ)).

    Algorithm 1 from the paper.
    """

    def __init__(self, wprf: WPRF, true_key: Optional[np.ndarray] = None,
                 verbose: bool = False, rng: Optional[np.random.Generator] = None):
        self.wprf = wprf
        self.true_key = true_key
        self.verbose = verbose
        self.rng = rng or np.random.default_rng()
        self._oracle_calls = 0

    # ------------------------------------------------------------------
    # Oracle access (counts every query)
    # ------------------------------------------------------------------

    def _oracle(self, x: np.ndarray) -> np.ndarray:
        """Call the wPRF oracle on input x using the (hidden) true key."""
        self._oracle_calls += 1
        return self.wprf.evaluate(self.true_key, x)

    # ------------------------------------------------------------------
    # Transition-point check  (Inequality 3 from the paper)
    # ------------------------------------------------------------------

    def _should_switch_to_exhaustive(self, c: int, H1: int) -> bool:
        """
        Return True when exhaustive search is cheaper than finding collision c+1.

        Inequality 3:
            Σ_{j=1}^{⌈λ/2^c⌉} C(H1, j)  <  2^{(λ+1)/2} · (√(c+1) - √c)

        where H1 is the current Hamming weight of K (our key estimate).
        """
        lam = self.wprf.params.lam
        threshold = 2 ** ((lam + 1) / 2) * (math.sqrt(c + 1) - math.sqrt(c))
        d_c = math.ceil(lam / (2 ** c))
        lhs = sum(math.comb(H1, j) for j in range(1, d_c + 1))
        return lhs < threshold

    # ------------------------------------------------------------------
    # Phase 1: Collision-based zero identification
    # ------------------------------------------------------------------

    def _find_collision(self, table: Dict[bytes, np.ndarray]
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Query random inputs until we find x' such that F(k, x') = F(k, x)
        for some x already in the table.  Returns (x, x') or None if budget
        exceeded.  Implements the inner 'repeat' loop of Algorithm 1.
        """
        max_iters = 10_000_000
        for _ in range(max_iters):
            x_new = self.wprf.sample_input(self.rng)
            y_new = self._oracle(x_new)
            key = _to_key(y_new)
            if key in table:
                x_old = table[key]
                if not np.array_equal(x_old, x_new):
                    return x_old, x_new
            table[key] = x_new
        return None

    def _update_key_estimate(self,
                              K: np.ndarray,
                              x: np.ndarray,
                              x_prime: np.ndarray,
                              hamming_dist_tracker: list) -> Tuple[np.ndarray, int]:
        """
        Given a collision (x, x'), update K by zeroing positions where x and
        x' differ.  Returns updated K and new Hamming weight H1.
        """
        diff_positions = np.where(x != x_prime)[0]
        for i in diff_positions:
            if K[i] == 1:
                K[i] = 0
        H1 = int(np.count_nonzero(K))

        if self.true_key is not None:
            hd = int(np.sum(K != self.true_key))
            hamming_dist_tracker.append(hd)

        return K, H1

    # ------------------------------------------------------------------
    # Phase 2: Exhaustive search
    # ------------------------------------------------------------------

    def _exhaustive_search(self,
                           K: np.ndarray,
                           d_c: int,
                           table: Dict[bytes, np.ndarray]) -> Optional[np.ndarray]:
        """
        Try all keys within one-sided Hamming distance d_c from K:
        only flip 1-bits to 0 (never 0 to 1, since we only ever add zeros).

        Verify each candidate against a stored (x, y) pair.
        Returns the correct key or None.
        """
        n = self.wprf.params.n
        one_positions = np.where(K == 1)[0]
        H1 = len(one_positions)
        d_c = min(d_c, H1)

        # Grab multiple verification pairs for reliable candidate checking.
        # Using log2(n)+4 pairs makes false positives negligible.
        import math as _math
        num_verify = max(4, _math.ceil(_math.log2(n + 1)) + 4)
        verify_pairs = list(table.items())[:num_verify]
        verify_data = [
            (np.frombuffer(yb, dtype=np.uint8).copy(), xv)
            for yb, xv in verify_pairs
        ]

        # Iterate over all subsets of 1-positions to flip
        from itertools import combinations

        def _check(k_candidate: np.ndarray) -> bool:
            for ref_y, ref_x in verify_data:
                self._oracle_calls += 1
                y_test = self.wprf.evaluate(k_candidate, ref_x)
                if not np.array_equal(y_test, ref_y):
                    return False
            return True

        # d_c = 0: just check K itself
        if _check(K.copy()):
            return K.copy()

        for flip_count in range(1, d_c + 1):
            for positions in combinations(range(H1), flip_count):
                k_candidate = K.copy()
                for pos in positions:
                    k_candidate[one_positions[pos]] = 0
                if _check(k_candidate):
                    return k_candidate

        return None   # not found at this d_c; caller will get more collisions

    # ------------------------------------------------------------------
    # Main attack entry point
    # ------------------------------------------------------------------

    def run(self, true_key: Optional[np.ndarray] = None) -> AttackResult:
        """
        Execute the full two-phase key recovery attack.

        Parameters
        ----------
        true_key : ndarray of shape (n,), optional
            If provided, overrides any key set at construction time.

        Returns
        -------
        AttackResult
        """
        if true_key is not None:
            self.true_key = true_key
        assert self.true_key is not None, "Must supply a true key."

        t0 = time.perf_counter()
        self._oracle_calls = 0
        lam = self.wprf.params.lam
        n = self.wprf.params.n

        # Initialise key estimate K = all-ones
        K = np.ones(n, dtype=np.int64)
        H1 = n
        c = 0
        hamming_trace = []
        table: Dict[bytes, np.ndarray] = {}

        # Seed the table with a few initial queries
        for _ in range(min(64, 2 ** (lam // 4))):
            x = self.wprf.sample_input(self.rng)
            y = self._oracle(x)
            table[_to_key(y)] = x

        q_col_end = 0   # oracle calls at end of collision phase

        while True:
            # --- Phase 1 inner loop: find next collision ---
            result = self._find_collision(table)
            if result is None:
                # Should not happen for well-chosen params; abort
                break
            x_old, x_new = result
            c += 1
            K, H1 = self._update_key_estimate(K, x_old, x_new, hamming_trace)

            if self.verbose:
                hd = int(np.sum(K != self.true_key)) if self.true_key is not None else "?"
                print(f"  Collision {c:3d}: H1={H1}, d(K,k)={hd}, "
                      f"oracle_calls={self._oracle_calls:,}")

            # --- Transition check: switch to exhaustive search? ---
            if self._should_switch_to_exhaustive(c, H1):
                q_col_end = self._oracle_calls
                d_c = math.ceil(lam / (2 ** c))
                found = self._exhaustive_search(K, d_c, table)
                if found is not None:
                    elapsed = time.perf_counter() - t0
                    return AttackResult(
                        success=True,
                        recovered_key=found,
                        true_key=self.true_key.copy(),
                        queries_collision_phase=q_col_end,
                        queries_exhaustive_phase=self._oracle_calls - q_col_end,
                        num_collisions=c,
                        hamming_distances=hamming_trace,
                        elapsed_seconds=elapsed,
                    )
                # Exhaustive search failed: continue collecting collisions
                if self.verbose:
                    print(f"  Exhaustive search at d_c={d_c} failed; continuing.")

        elapsed = time.perf_counter() - t0
        return AttackResult(
            success=False,
            recovered_key=None,
            true_key=self.true_key.copy(),
            queries_collision_phase=self._oracle_calls,
            queries_exhaustive_phase=0,
            num_collisions=c,
            hamming_distances=hamming_trace,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Reversed (F3, F2) One-to-One attack
# ---------------------------------------------------------------------------

class ReversedAttack:
    """
    Key recovery attack on the Reversed (F3, F2)-wPRF with One-to-One params.

    Structural difference from the standard attack
    -----------------------------------------------
    The key k ∈ {0,1,2}^n (ternary).  Positions where k_i = 0 still kill the
    corresponding input component via k_i ⊙_3 x_i = 0.

    Phase 1: Collision-based identification of zero positions.
    - im(F) has size 3^{h1^* + h2^*} ≈ 2^{4λ/3} (both non-zero value counts).
    - Each collision reveals ≈ 2/3 of remaining unknown zero positions.
    - Repeat until all zero positions J_0 are identified.

    Phase 2: Exhaustive search over non-zero positions J_1 ∪ J_2.
    - Each non-zero position has exactly 2 possible values {1, 2}.
    - |J_1 ∪ J_2| ≈ (2/3) · n  →  2^{(2/3)·n} ≈ 2^{0.84λ} candidates.
    - Each candidate verified with one oracle query.

    Total: O(2^{0.84λ}).
    """

    def __init__(self, wprf: WPRF, true_key: Optional[np.ndarray] = None,
                 verbose: bool = False, rng: Optional[np.random.Generator] = None):
        self.wprf = wprf
        self.true_key = true_key
        self.verbose = verbose
        self.rng = rng or np.random.default_rng()
        self._oracle_calls = 0

    def _oracle(self, x: np.ndarray) -> np.ndarray:
        self._oracle_calls += 1
        return self.wprf.evaluate(self.true_key, x)

    # ------------------------------------------------------------------
    # Phase 1: Identify zero positions via collisions
    # ------------------------------------------------------------------

    def _find_collision(self, table: Dict[bytes, np.ndarray]
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Find a collision in the output table."""
        for _ in range(10_000_000):
            x_new = self.wprf.sample_input(self.rng)
            y_new = self._oracle(x_new)
            key = _to_key(y_new)
            if key in table and not np.array_equal(table[key], x_new):
                return table[key], x_new
            table[key] = x_new
        return None

    def _identify_zero_positions(self) -> Tuple[np.ndarray, int]:
        """
        Accumulate collisions until all zero positions in k are likely found.
        Safety margin: run 3× the theoretical minimum number of collisions.

        Returns (J0_mask, oracle_calls_used) where J0_mask[i]=True means
        position i is identified as k_i = 0.
        """
        n = self.wprf.params.n
        lam = self.wprf.params.lam
        h0_star = n // 3   # expected number of zeros in a ternary key

        # Minimum collisions needed (solved numerically from the paper's eq.)
        # (1 - Σ_{i=1}^{c} 2/3^i) · h0^* < 1  → c ≈ log_{3/2}(h0^*)
        c_min = max(1, math.ceil(math.log(h0_star + 1) / math.log(1.5)))
        c_target = 3 * c_min   # safety margin

        J0_mask = np.zeros(n, dtype=bool)
        table: Dict[bytes, np.ndarray] = {}
        c = 0

        while c < c_target:
            result = self._find_collision(table)
            if result is None:
                break
            x_old, x_new = result
            diff = np.where(x_old != x_new)[0]
            J0_mask[diff] = True
            c += 1

            if self.verbose:
                known_zeros = int(np.sum(J0_mask))
                print(f"  Rev collision {c}: diff_positions={len(diff)}, "
                      f"identified zeros={known_zeros}/{n}")

        return J0_mask, table

    # ------------------------------------------------------------------
    # Phase 2: Exhaustive search over non-zero positions
    # ------------------------------------------------------------------

    def _exhaustive_search(self,
                           J0_mask: np.ndarray,
                           table: Dict[bytes, np.ndarray]) -> Optional[np.ndarray]:
        """
        Enumerate all assignments {1, 2}^{|non-zero positions|}.
        Verify each against a stored (x, y) pair.
        """
        n = self.wprf.params.n
        non_zero_positions = np.where(~J0_mask)[0]
        num_free = len(non_zero_positions)

        if self.verbose:
            print(f"  Exhaustive search: {num_free} free positions "
                  f"→ 2^{num_free} = {2**num_free:,} candidates")

        ref_y_bytes, ref_x = next(iter(table.items()))
        ref_y = np.frombuffer(ref_y_bytes, dtype=np.uint8).copy()

        # Build base candidate (all zeros at free positions = 1, others 0)
        base = np.zeros(n, dtype=np.int64)

        # Iterate over all 2^{num_free} assignments
        for mask in range(2 ** num_free):
            k_candidate = base.copy()
            for bit_idx, pos in enumerate(non_zero_positions):
                # bit 0 → value 1, bit 1 → value 2
                k_candidate[pos] = 1 + ((mask >> bit_idx) & 1)

            self._oracle_calls += 1
            y_test = self.wprf.evaluate(k_candidate, ref_x)
            if np.array_equal(y_test, ref_y):
                return k_candidate

        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, true_key: Optional[np.ndarray] = None) -> AttackResult:
        if true_key is not None:
            self.true_key = true_key
        assert self.true_key is not None

        t0 = time.perf_counter()
        self._oracle_calls = 0

        # Phase 1
        J0_mask, table = self._identify_zero_positions()
        q_col_end = self._oracle_calls

        # Phase 2
        found = self._exhaustive_search(J0_mask, table)

        elapsed = time.perf_counter() - t0
        return AttackResult(
            success=found is not None,
            recovered_key=found,
            true_key=self.true_key.copy(),
            queries_collision_phase=q_col_end,
            queries_exhaustive_phase=self._oracle_calls - q_col_end,
            num_collisions=int(np.sum(J0_mask)),
            elapsed_seconds=elapsed,
        )
