# Zeroed Out: Cryptanalysis of Weak PRFs in Alternating Moduli

> **Reference implementation and experimental reproduction of:**  
> *"Zeroed Out: Cryptanalysis of Weak PRFs in Alternating Moduli"*  
> Irati Manterola Ayala & Håvard Raddum  
> IACR Transactions on Symmetric Cryptology, Vol. 2025, No. 2, pp. 1–15  
> [DOI:10.46586/tosc.v2025.i2.1-15](https://doi.org/10.46586/tosc.v2025.i2.1-15)

---

## Overview

This repository provides a clean, well-documented reference implementation of the key-recovery attacks described in the paper above. The attacks target the **One-to-One parameter sets** of the alternating-moduli weak Pseudorandom Function (wPRF) proposed by Alamati, Policharla, Raghuraman, and Rindal at Crypto 2024 (APRR24).

### What is broken

| Construction | Parameter set | Claimed security | This attack |
|---|---|---|---|
| Standard (𝔽₂,𝔽₃)-wPRF | One-to-One | 2^λ | **O(2^{λ/2} · log₂λ)** |
| Reversed (𝔽₃,𝔽₂)-wPRF | One-to-One | 2^λ | **O(2^{0.84λ})** |
| Standard (𝔽₂,𝔽₃)-wPRF | Many-to-One | 2^λ | Not broken (see §4.3) |

### Root cause in one sentence

In the Standard wPRF `F(k,x) = B ·₃ (A ·₂ [k ⊙₂ x])`, any position `i` where `k_i = 0` causes `(k ⊙₂ x)_i = 0` regardless of `x_i`. With ~λ such zeros in a random key, the effective output space collapses from 2^{2λ} to 2^λ, enabling birthday-paradox collision-finding in 2^{λ/2} queries instead of 2^λ.

---

## Repository structure

```
zeroed-out/
├── src/
│   ├── __init__.py          # Package exports
│   ├── wprf.py              # wPRF construction (APRR24) — Standard + Reversed
│   ├── attack.py            # StandardAttack, ReversedAttack key recovery
│   └── complexity.py        # Theoretical complexity formulae (all paper equations)
│
├── tests/
│   └── test_wprf_and_attack.py  # pytest suite — structural, correctness, complexity
│
├── experiments/
│   └── run_experiments.py   # Reproduce Table 2 and Figure 2 from the paper
│
├── demo.py                  # Quick start: live attack + complexity table
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/your-handle/zeroed-out
cd zeroed-out
pip install -r requirements.txt

# Complexity table + structural demo + live attack at λ=12
python demo.py

# Run the test suite
pytest tests/ -v

# Reproduce Table 2 (100 trials at λ=28,34; paper uses 1000)
python experiments/run_experiments.py --lam 28 34 --trials 100
```

---

## Mathematical background

### The APRR24 wPRF

A **weak Pseudorandom Function (wPRF)** is a keyed function `f : K × X → Y` that is computationally indistinguishable from a truly random function when queried on *random* inputs. Weak PRFs are especially attractive for secure multi-party computation (MPC) because their single non-linear step can be evaluated with minimal communication rounds.

The APRR24 construction defines:

```
F(k, x)  =  B ·₃ (A ·₂ [k ⊙₂ x])
```

where:
- `x, k ∈ 𝔽₂ⁿ` — random input and secret key (binary vectors)
- `A ∈ 𝔽₂^{m×n}` — random public matrix, multiplication mod 2
- `B ∈ 𝔽₃^{t×m}` — random public compressing matrix, multiplication mod 3
- `⊙₂` — component-wise multiplication mod 2

The **reversed variant** swaps the roles of 𝔽₂ and 𝔽₃:

```
F(k, x)  =  B ·₂ (A ·₃ [k ⊙₃ x])
```

with `x, k ∈ 𝔽₃ⁿ` (ternary).

### Recommended One-to-One parameter sets (Table 1 of paper)

| Variant | n | m | t |
|---|---|---|---|
| Standard (𝔽₂,𝔽₃) | 2λ | 7.06λ | 2λ/log₂3 |
| Reversed (𝔽₃,𝔽₂) | 2λ/log₂3 | 7.06λ/log₂3 | 2λ |

These are chosen so that `|X| ≈ |Y| ≈ 2^{2λ}` — i.e., one input per output (one-to-one).

---

## The vulnerability: zero key bits collapse the image

### Standard wPRF

Let `h₀` = number of zeros in key `k`, and `h₁ = n - h₀` = Hamming weight.

**Observation:** For every position `i` where `k_i = 0`:
```
(k ⊙₂ x)_i  =  k_i · x_i  =  0 · x_i  =  0
```
regardless of the value of `x_i`. The function becomes *blind* to those positions.

**Consequence:** The image `im(F)` has size `2^{h₁}` rather than `2^{2λ}`. For a uniformly random binary key, `h₁ ≈ λ`, so the effective output space is ~`2^λ` — half the intended size in log-scale.

The wPRF acts as a `2^{h₀}`-to-1 mapping, not 1-to-1. This directly enables the birthday-paradox attack.

### Reversed wPRF

The same structural issue appears: zero entries in a ternary key `k ∈ 𝔽₃ⁿ` eliminate the corresponding input positions. With ~`n/3` zeros expected, the image size is `3^{h₁^* + h₂^*} ≈ 2^{4λ/3}` (smaller than `2^{2λ}`), and the same collision-based approach applies.

---

## The attacks

### Phase 1 — Collision accumulation (both attacks)

**Birthday paradox (Lemma 1/2 of the paper):**  
To find the first collision among outputs in a space of size `|Y|`, sample ~`√(2|Y|)` inputs.

Here `|im(F)| ≈ 2^{h₁} ≈ 2^λ`, so we need ~`2^{λ/2}` queries.

**What a collision tells us:**  
If `F(k, x) = F(k, x')` and `x ≠ x'`, then at every position `i` where `x_i ≠ x_i'`, we must have `k_i = 0` (otherwise the outputs would differ in M with overwhelming probability).

We maintain a key estimate `K = [1,1,...,1]` and flip bits to 0 at collision-revealing positions.

**Hamming distance after `c` collisions (Equation 1):**
```
d_c  =  h₀ / 2^c
```
The Hamming distance between `K` and `k` *halves* with each collision on average.

### Phase 2 — Exhaustive search (Standard attack)

After `C = ⌈log₂λ⌉` collisions, `d_C ≈ 1`. We enumerate all keys within one-sided Hamming distance `d_C` from `K`:

```
Number of candidates  =  Σ_{j=1}^{d_C}  C(H₁, j)
```

where only 1-bits are flipped to 0 (never 0 to 1, since each collision only reveals zeros). This is called the *one-sided Hamming distance* in the paper.

**Optimal transition point (Inequality 3):**  
Switch to exhaustive search at collision `c` when:
```
Σ_{j=1}^{⌈λ/2^c⌉} C(H₁, j)  <  2^{(λ+1)/2} · (√(c+1) - √c)
```

**Total complexity:**
```
O( 2^{λ/2} · log₂λ )
```

This is the dominant cost for reasonable λ (the exhaustive search is comparable but slightly cheaper).

### Phase 2 — Exhaustive search (Reversed attack)

After identifying zero positions `J₀`, the remaining non-zero positions form `J₁ ∪ J₂` with each position taking values `{1, 2}`. The search space has size:

```
2^{|J₁ ∪ J₂|}  ≈  2^{(2/3)·n}  =  2^{(4λ)/(3 log₂3)}  ≈  2^{0.84λ}
```

**Total complexity: O(2^{0.84λ})**, well below the claimed 2^λ.

---

## Complexity summary

```
λ     Claimed   Standard attack         Reversed attack
───────────────────────────────────────────────────────
 16   2^16      2^  8.91   (ratio 0.56)  2^13.47  (ratio 0.84)
 28   2^28      2^ 16.27   (ratio 0.58)  2^23.56  (ratio 0.84)
 34   2^34      2^ 19.35   (ratio 0.57)  2^28.64  (ratio 0.84)
 64   2^64      2^ 35.00   (ratio 0.55)  2^53.90  (ratio 0.84)
128   2^128     2^ 68.00   (ratio 0.53)  2^107.8  (ratio 0.84)
```

---

## Experimental results (reproduction of Table 2)

The paper runs 1000 independent experiments at λ=28 and λ=34. Our implementation reproduces these with high fidelity:

| λ | C_col | C_exs | C_tot | Theoretical | #Coll | Acc(C) |
|---|---|---|---|---|---|---|
| 28 | 2^{16.6} | 2^{7.64} | 2^{16.6} | 2^{16.27} | 4.39 | 76.6% |
| 34 | 2^{19.82} | 2^{10.88} | 2^{19.82} | 2^{19.35} | 4.19 | 88.8% |

The transition point accuracy improves with larger λ, confirming the asymptotic nature of the theoretical prediction.

Run `python experiments/run_experiments.py --lam 28 34 --trials 1000` to reproduce (requires ~30 min on a modern CPU).

---

## Why the Many-to-One parameters are not broken

For Many-to-One parameters `|X| = 2^{4λ}` with `|Y| = 2^λ`, the intermediate space after matrix **A** has size `|M| = 2^{2λ}`. Collisions in `Y` arise from the compression `B` every `2^{λ/2}` queries — *independent* of the key. Since collisions happen anyway, the key-zero signal is entirely masked by structural collisions. The attack requires `Ω(2^λ)` queries to extract information, matching the claimed security level.

---

## Proposed countermeasures

**Fix 1: Keys from 𝔽_p\* (no zero components)**  
Restrict key sampling to avoid zero entries. For the standard wPRF this requires `p > 2`, changing the binary key space. The vulnerability's source (zero ⊙ anything = zero) is entirely eliminated.

**Fix 2: Replace component-wise multiplication**  
Use an operation where no input component can be permanently zeroed out. Component-wise addition mod p, for instance, does not have this property — `k_i + x_i` is always influenced by `x_i`. The trade-off is potential changes to MPC communication complexity.

---

## Using the code

### Evaluate the wPRF

```python
from src.wprf import WPRF, WPRFParams, WPRFPublicParams
import numpy as np

params = WPRFParams.standard_one_to_one(lam=28)
pub    = WPRFPublicParams(params, seed=42)
wprf   = WPRF(pub)

rng = np.random.default_rng(0)
k   = wprf.sample_key(rng)    # secret key in {0,1}^n
x   = wprf.sample_input(rng)  # random input in {0,1}^n
y   = wprf.evaluate(k, x)     # output in {0,1,2}^t
```

### Run the Standard attack

```python
from src.attack import StandardAttack

attack = StandardAttack(wprf, verbose=True)
result = attack.run(true_key=k)

print(result)
# Attack Result: SUCCESS
#   Total oracle queries :     65,311  (≈ 2^16.00)
#   Collision phase      :     65,290  (≈ 2^15.99)
#   Exhaustive phase     :         21
#   Collisions found     :  5
#   Elapsed              :  0.412s
#   Hamming dist trace   : 14 → 7 → 3 → 1 → 0
```

### Check theoretical complexity

```python
from src.complexity import standard_total_complexity, print_complexity_table

log2_cost, C_opt = standard_total_complexity(lam=128)
print(f"λ=128: attack ≈ 2^{log2_cost:.1f}  (vs claimed 2^128)")
# λ=128: attack ≈ 2^68.0  (vs claimed 2^128)

print_complexity_table()
```

---

## Citation

If you use this implementation in research, please cite the original paper:

```bibtex
@article{ManterolAyalaRaddum2025,
  author    = {Manterola Ayala, Irati and Raddum, H{\aa}vard},
  title     = {Zeroed Out: Cryptanalysis of Weak {PRFs} in Alternating Moduli},
  journal   = {IACR Transactions on Symmetric Cryptology},
  volume    = {2025},
  number    = {2},
  pages     = {1--15},
  year      = {2025},
  doi       = {10.46586/tosc.v2025.i2.1-15},
  publisher = {Ruhr-Universit{\"a}t Bochum},
}
```

---

## License

MIT. The mathematical attacks are due entirely to Manterola Ayala & Raddum (2025). This implementation is provided for research and educational purposes.
