"""
tests/test_wprf_and_attack.py
==============================
pytest test suite for the Zeroed Out implementation.
Covers:
  - Correctness and determinism of the wPRF
  - Structural properties (image size, key structure)
  - Attack correctness at small λ
  - Complexity formula sanity
"""

import math
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.wprf import WPRF, WPRFParams, WPRFPublicParams
from src.attack import StandardAttack, ReversedAttack
from src.complexity import (
    standard_total_complexity,
    reversed_total_complexity,
    hamming_distance_after_collisions,
    birthday_samples,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def small_std_params():
    """Small Standard (F2,F3) wPRF for fast unit tests (λ=8)."""
    return WPRFParams.standard_one_to_one(lam=8)


@pytest.fixture
def small_rev_params():
    """Small Reversed (F3,F2) wPRF for fast unit tests (λ=8)."""
    return WPRFParams.reversed_one_to_one(lam=8)


@pytest.fixture
def wprf_std(small_std_params):
    pub = WPRFPublicParams(small_std_params, seed=42)
    return WPRF(pub)


@pytest.fixture
def wprf_rev(small_rev_params):
    pub = WPRFPublicParams(small_rev_params, seed=42)
    return WPRF(pub)


# ===========================================================================
# wPRF construction tests
# ===========================================================================

class TestWPRFParams:

    def test_standard_param_dimensions(self):
        for lam in [8, 16, 20]:
            p = WPRFParams.standard_one_to_one(lam)
            assert p.n == 2 * lam
            assert p.m == round(7.06 * lam)
            assert p.t == round(2 * lam / math.log2(3))
            assert p.p == 2
            assert p.q == 3

    def test_reversed_param_dimensions(self):
        for lam in [8, 16]:
            p = WPRFParams.reversed_one_to_one(lam)
            assert p.p == 3
            assert p.q == 2
            # Input space (F3^n) should ≈ output space (F2^t) ≈ 2^{2λ}
            assert abs(p.input_space_bits - 2 * lam) < 2
            assert abs(p.output_space_bits - 2 * lam) < 2

    def test_many_to_one_larger_input(self):
        for lam in [8, 16]:
            p = WPRFParams.standard_many_to_one(lam)
            assert p.input_space_bits > p.output_space_bits


class TestWPRFEvaluation:

    def test_output_shape(self, wprf_std):
        rng = np.random.default_rng(0)
        k = wprf_std.sample_key(rng)
        x = wprf_std.sample_input(rng)
        y = wprf_std.evaluate(k, x)
        assert y.shape == (wprf_std.params.t,)

    def test_output_alphabet_standard(self, wprf_std):
        rng = np.random.default_rng(0)
        k = wprf_std.sample_key(rng)
        for _ in range(50):
            x = wprf_std.sample_input(rng)
            y = wprf_std.evaluate(k, x)
            assert np.all(y >= 0) and np.all(y < 3), "Standard wPRF output must be in {0,1,2}"

    def test_output_alphabet_reversed(self, wprf_rev):
        rng = np.random.default_rng(0)
        k = wprf_rev.sample_key(rng)
        for _ in range(50):
            x = wprf_rev.sample_input(rng)
            y = wprf_rev.evaluate(k, x)
            assert np.all(y >= 0) and np.all(y < 2), "Reversed wPRF output must be in {0,1}"

    def test_determinism(self, wprf_std):
        """Same (k, x) must always give same output."""
        rng = np.random.default_rng(99)
        k = wprf_std.sample_key(rng)
        x = wprf_std.sample_input(rng)
        y1 = wprf_std.evaluate(k, x)
        y2 = wprf_std.evaluate(k, x)
        assert np.array_equal(y1, y2)

    def test_key_sensitivity(self, wprf_std):
        """Different keys should (almost always) give different outputs."""
        rng = np.random.default_rng(7)
        x = wprf_std.sample_input(rng)
        outputs = set()
        for _ in range(20):
            k = wprf_std.sample_key(rng)
            y = wprf_std.evaluate(k, x)
            outputs.add(y.tobytes())
        assert len(outputs) > 1, "All 20 distinct keys gave the same output — suspicious"

    def test_batch_matches_single(self, wprf_std):
        """Batch evaluation must match single-input evaluation."""
        rng = np.random.default_rng(13)
        k = wprf_std.sample_key(rng)
        X = np.array([wprf_std.sample_input(rng) for _ in range(30)])
        Y_batch = wprf_std.evaluate_batch(k, X)
        for i, x in enumerate(X):
            y_single = wprf_std.evaluate(k, x)
            assert np.array_equal(Y_batch[i], y_single), f"Batch mismatch at index {i}"


class TestStructuralVulnerability:

    def test_zero_key_position_makes_input_irrelevant(self, wprf_std):
        """
        Core vulnerability: if k[i] = 0, then x[i] has no effect on F(k,x).
        """
        rng = np.random.default_rng(42)
        k = wprf_std.sample_key(rng)
        k[0] = 0   # force position 0 to zero

        x = wprf_std.sample_input(rng)
        x_flip = x.copy()
        x_flip[0] ^= 1  # flip position 0

        y1 = wprf_std.evaluate(k, x)
        y2 = wprf_std.evaluate(k, x_flip)
        assert np.array_equal(y1, y2), (
            "Flipping a zero-key-position should not change the output"
        )

    def test_image_shrinks_with_zero_key_bits(self, wprf_std):
        """
        A key with many zeros should have a smaller effective image than
        a key with few zeros — confirming the reduced output space.
        """
        rng = np.random.default_rng(55)
        n = wprf_std.params.n

        # Heavy zero key (h0 ≈ 3n/4)
        k_many_zeros = wprf_std.sample_key(rng)
        k_many_zeros[:n * 3 // 4] = 0

        # Light zero key (h0 ≈ n/4)
        k_few_zeros = np.ones(n, dtype=np.int64)
        k_few_zeros[:n // 4] = 0

        num_samples = 500
        im_many = set()
        im_few = set()
        for _ in range(num_samples):
            x = wprf_std.sample_input(rng)
            im_many.add(wprf_std.evaluate(k_many_zeros, x).tobytes())
            im_few.add(wprf_std.evaluate(k_few_zeros, x).tobytes())

        assert len(im_many) < len(im_few), (
            "Key with more zeros should produce a smaller observed image"
        )

    def test_collision_reveals_zero_positions(self, wprf_std):
        """
        When F(k, x) = F(k, x') with x ≠ x', positions where x_i ≠ x_i'
        must have k_i = 0.  Verify this empirically.
        """
        rng = np.random.default_rng(77)
        k = wprf_std.sample_key(rng)

        # Force known collision: flip only zero-key positions
        zero_pos = wprf_std.zero_positions(k)
        assert len(zero_pos) > 0, "Need at least one zero bit in key for this test"

        x = wprf_std.sample_input(rng)
        x_prime = x.copy()
        x_prime[zero_pos[0]] ^= 1  # flip only at a zero-key position

        y  = wprf_std.evaluate(k, x)
        y2 = wprf_std.evaluate(k, x_prime)
        assert np.array_equal(y, y2), "Flip at zero-key position should not change output"

        # Confirm flipping a one-key position DOES change output (usually)
        one_pos = np.where(k == 1)[0]
        if len(one_pos) > 0:
            x_prime2 = x.copy()
            x_prime2[one_pos[0]] ^= 1
            y3 = wprf_std.evaluate(k, x_prime2)
            # This can rarely equal y by coincidence; just don't assert equality


# ===========================================================================
# Attack tests  (small λ only — for speed)
# ===========================================================================

class TestStandardAttack:

    @pytest.mark.parametrize("lam", [8, 10, 12])
    def test_recovers_key(self, lam):
        """Attack must recover the correct key for small λ."""
        params = WPRFParams.standard_one_to_one(lam)
        pub = WPRFPublicParams(params, seed=lam * 17)
        wprf = WPRF(pub)

        rng = np.random.default_rng(lam)
        true_key = wprf.sample_key(rng)

        attack = StandardAttack(wprf, verbose=False, rng=np.random.default_rng(lam + 1))
        result = attack.run(true_key=true_key)

        assert result.success, f"Attack failed for λ={lam}"
        assert np.array_equal(result.recovered_key, true_key), (
            f"Recovered key differs from true key at λ={lam}"
        )

    def test_query_complexity_within_theoretical_bound(self):
        """
        For λ=10, verify that total queries ≤ 10 × theoretical prediction
        (generous bound — empirical variance is expected at small λ).
        """
        lam = 10
        theoretical_log2, _ = standard_total_complexity(lam)
        theoretical_bound = 10 * (2 ** theoretical_log2)

        params = WPRFParams.standard_one_to_one(lam)
        pub = WPRFPublicParams(params, seed=999)
        wprf = WPRF(pub)
        rng_k = np.random.default_rng(42)
        true_key = wprf.sample_key(rng_k)

        attack = StandardAttack(wprf, verbose=False, rng=np.random.default_rng(43))
        result = attack.run(true_key=true_key)

        assert result.success
        assert result.total_queries <= theoretical_bound, (
            f"Used {result.total_queries} queries, expected ≤ {theoretical_bound:.0f} "
            f"(10× theoretical 2^{theoretical_log2:.1f})"
        )

    def test_hamming_distance_decreases(self):
        """
        Verify that the Hamming distance between key estimate and true key
        decreases (not necessarily monotonically) as collisions accumulate.
        """
        lam = 10
        params = WPRFParams.standard_one_to_one(lam)
        pub = WPRFPublicParams(params, seed=7)
        wprf = WPRF(pub)
        rng = np.random.default_rng(8)
        true_key = wprf.sample_key(rng)

        attack = StandardAttack(wprf, verbose=False, rng=np.random.default_rng(9))
        result = attack.run(true_key=true_key)

        if result.hamming_distances and len(result.hamming_distances) >= 2:
            # The Hamming distance should overall decrease
            first_half  = result.hamming_distances[:len(result.hamming_distances)//2]
            second_half = result.hamming_distances[len(result.hamming_distances)//2:]
            assert max(second_half, default=0) <= max(first_half, default=float('inf')), (
                "Hamming distance should not increase overall"
            )


# ===========================================================================
# Complexity formula tests
# ===========================================================================

class TestComplexityFormulae:

    def test_standard_complexity_below_claimed(self):
        """Attack complexity must be strictly below claimed λ-bit security."""
        for lam in [16, 20, 24, 28, 32]:
            log2_attack, _ = standard_total_complexity(lam)
            assert log2_attack < lam, (
                f"Standard attack at λ={lam}: complexity 2^{log2_attack:.2f} "
                f"should be < 2^{lam} (claimed security)"
            )

    def test_reversed_complexity_below_claimed(self):
        """Reversed attack complexity must be strictly below claimed security."""
        for lam in [16, 20, 24, 28, 32]:
            log2_attack = reversed_total_complexity(lam)
            assert log2_attack < lam, (
                f"Reversed attack at λ={lam}: complexity 2^{log2_attack:.2f} "
                f"should be < 2^{lam}"
            )

    def test_standard_complexity_approx_half_lambda(self):
        """Standard attack should be ≈ 2^{λ/2} (within a small constant factor)."""
        for lam in [20, 28, 34, 64]:
            log2_attack, _ = standard_total_complexity(lam)
            ratio = log2_attack / lam
            assert 0.45 < ratio < 0.65, (
                f"Standard complexity ratio at λ={lam}: {ratio:.3f}, expected ≈ 0.5"
            )

    def test_reversed_complexity_approx_0_84_lambda(self):
        """Reversed attack should be ≈ 2^{0.84λ}."""
        for lam in [20, 28, 34, 64]:
            log2_attack = reversed_total_complexity(lam)
            ratio = log2_attack / lam
            assert 0.78 < ratio < 0.90, (
                f"Reversed complexity ratio at λ={lam}: {ratio:.3f}, expected ≈ 0.84"
            )

    def test_hamming_halving(self):
        """Equation (1): d_c = h0 / 2^c."""
        h0 = 100
        for c in range(1, 8):
            d = hamming_distance_after_collisions(h0, c)
            expected = h0 / (2 ** c)
            assert abs(d - expected) < 1e-9

    def test_birthday_bound(self):
        """Birthday paradox: ≈ √(2|Y|) samples for first collision."""
        for bits in [8, 12, 16, 20]:
            log2_samples = birthday_samples(bits, num_collisions=1)
            # Should be (bits + 1) / 2
            expected = (bits + 1) / 2
            assert abs(log2_samples - expected) < 1e-9


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
