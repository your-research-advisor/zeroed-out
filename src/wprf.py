"""
wprf.py
=======
Reference implementation of the alternating-moduli weak Pseudorandom Functions
from Alamati, Policharla, Raghuraman, Rindal (APRR24), presented at Crypto 2024.

Two constructions are implemented:
  - Standard  (F2, F3)-wPRF  :  F(k,x) = B ·₃ (A ·₂ [k ⊙₂ x])
  - Reversed  (F3, F2)-wPRF  :  F(k,x) = B ·₂ (A ·₃ [k ⊙₃ x])

Mathematical notation
---------------------
  ⊙_p   component-wise multiplication mod p
  ·_p    matrix–vector multiplication mod p
  F_p    the field / ring of integers mod p
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter sets (Table 1 of the paper)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WPRFParams:
    """
    A frozen parameter set for the APRR24 wPRF.

    Attributes
    ----------
    lam : int
        Security parameter λ (in bits).
    n : int
        Input / key dimension (over F_p).
    m : int
        Intermediate dimension (after first linear map).
    t : int
        Output dimension (after compressing matrix B).
    p : int
        First modulus (inner domain).
    q : int
        Second modulus (outer domain).
    name : str
        Human-readable label.
    """
    lam: int
    n: int
    m: int
    t: int
    p: int
    q: int
    name: str

    @classmethod
    def standard_one_to_one(cls, lam: int) -> "WPRFParams":
        """
        Standard (F2, F3)-wPRF, One-to-One parameter set.
        n = 2λ,  m = 7.06λ,  t = 2λ / log2(3)
        Claimed security: λ bits.
        """
        import math
        n = 2 * lam
        m = round(7.06 * lam)
        t = round(2 * lam / math.log2(3))
        return cls(lam=lam, n=n, m=m, t=t, p=2, q=3,
                   name=f"Standard-121-λ{lam}")

    @classmethod
    def reversed_one_to_one(cls, lam: int) -> "WPRFParams":
        """
        Reversed (F3, F2)-wPRF, One-to-One parameter set.
        n = 2λ / log2(3),  m = 7.06λ / log2(3),  t = 2λ
        Claimed security: λ bits.
        """
        import math
        log23 = math.log2(3)
        n = round(2 * lam / log23)
        m = round(7.06 * lam / log23)
        t = 2 * lam
        return cls(lam=lam, n=n, m=m, t=t, p=3, q=2,
                   name=f"Reversed-121-λ{lam}")

    @classmethod
    def standard_many_to_one(cls, lam: int) -> "WPRFParams":
        """
        Standard (F2, F3)-wPRF, Many-to-One parameter set.
        n = 4λ,  m = 2λ,  t = λ / log2(3)
        Claimed security: λ bits.
        """
        import math
        n = 4 * lam
        m = 2 * lam
        t = round(lam / math.log2(3))
        return cls(lam=lam, n=n, m=m, t=t, p=2, q=3,
                   name=f"Standard-M21-λ{lam}")

    @property
    def input_space_bits(self) -> int:
        """log2 of the input space size."""
        import math
        return round(self.n * math.log2(self.p))

    @property
    def output_space_bits(self) -> int:
        """log2 of the output space size."""
        import math
        return round(self.t * math.log2(self.q))


# ---------------------------------------------------------------------------
# Public matrices (fixed per evaluation context)
# ---------------------------------------------------------------------------

class WPRFPublicParams:
    """
    Random public matrices A and B sampled once and reused across evaluations.
    In a real protocol these would be agreed upon by all parties.
    """

    def __init__(self, params: WPRFParams, seed: Optional[int] = None):
        self.params = params
        rng = np.random.default_rng(seed)

        # A ∈ F_p^{m×n}
        self.A = rng.integers(0, params.p, size=(params.m, params.n))

        # B ∈ F_q^{t×m}
        self.B = rng.integers(0, params.q, size=(params.t, params.m))


# ---------------------------------------------------------------------------
# wPRF evaluation
# ---------------------------------------------------------------------------

class WPRF:
    """
    Evaluates the APRR24 (F_p, F_q)-wPRF:

        F(k, x) = B ·_q  ( A ·_p  [k ⊙_p x] )

    For the reversed variant (F_q, F_p) the roles of p/q are swapped:

        F(k, x) = B ·_p  ( A ·_q  [k ⊙_q x] )

    The code treats the standard case uniformly; the reversed case is handled
    by the caller constructing params with p=3, q=2.
    """

    def __init__(self, pub: WPRFPublicParams):
        self.pub = pub
        self.params = pub.params

    # ------------------------------------------------------------------
    # Key and input sampling
    # ------------------------------------------------------------------

    def sample_key(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample a uniformly random key k ∈ F_p^n."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, self.params.p, size=self.params.n)

    def sample_input(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample a uniformly random input x ∈ F_p^n."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, self.params.p, size=self.params.n)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, k: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute F(k, x).

        Steps
        -----
        1.  v = k ⊙_p x          (component-wise multiplication mod p)
        2.  w = A ·_p v           (matrix–vector multiply mod p)
        3.  y = B ·_q w           (matrix–vector multiply mod q, after embedding)
        """
        p, q = self.params.p, self.params.q
        A, B = self.pub.A, self.pub.B

        # Step 1: pointwise multiply mod p
        v = (k * x) % p                          # shape (n,)

        # Step 2: linear map mod p
        w = A @ v % p                             # shape (m,)

        # Step 3: natural embedding F_p → F_q is identity on {0,...,p-1}
        #         then linear map mod q
        y = B @ w % q                             # shape (t,)
        return y

    def evaluate_batch(self, k: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Vectorised evaluation of F(k, x) for a batch of inputs.

        Parameters
        ----------
        k : ndarray of shape (n,)
        X : ndarray of shape (batch, n)

        Returns
        -------
        Y : ndarray of shape (batch, t)
        """
        p, q = self.params.p, self.params.q
        A, B = self.pub.A, self.pub.B

        V = (X * k[np.newaxis, :]) % p           # (batch, n)
        W = (V @ A.T) % p                         # (batch, m)
        Y = (W @ B.T) % q                         # (batch, t)
        return Y

    # ------------------------------------------------------------------
    # Helpers for analysis
    # ------------------------------------------------------------------

    def hamming_weight(self, k: np.ndarray) -> int:
        """Number of non-zero entries in k."""
        return int(np.count_nonzero(k))

    def zero_positions(self, k: np.ndarray) -> np.ndarray:
        """Indices where k_i == 0."""
        return np.where(k == 0)[0]

    def effective_image_size_bits(self, k: np.ndarray) -> float:
        """
        Theoretical log2 of |im(F)| given key k.
        For standard wPRF: 2^{h1} where h1 = hamming_weight(k).
        For reversed wPRF: 3^{h1^* + h2^*} where h_i* = count of value i in k.
        """
        import math
        p = self.params.p
        if p == 2:
            h1 = self.hamming_weight(k)
            return float(h1)
        else:  # p == 3, reversed moduli
            h1 = int(np.sum(k == 1))
            h2 = int(np.sum(k == 2))
            return (h1 + h2) * math.log2(3)
