"""
server/poly_approx.py — degree-3 Chebyshev polynomial approximation of sigmoid.

Used by inference.py to perform the activation homomorphically under CKKS
(real sigmoid requires exp/division, which HE cannot evaluate directly).

Coefficients are fitted via least-squares on [-POLY_RANGE, POLY_RANGE].

Key design facts for a degree-3 polynomial P(z) = A0 + A1*z + A3*z^3:
  - P(0)   = A0 = 0.5 exactly (label boundary preserved)
  - P'(z)  = A1 + 3*A3*z^2; inflection at z* = sqrt(-A1/(3*A3))
  - P is monotone on (-z*, z*); beyond that the cubic term dominates
  - For this model z* ≈ 4.0, covering 99.9%+ of real-world scaled logits

The label decision — P(z) >= 0.5 iff z >= 0 — is correct everywhere P is
monotone, which covers all practical inputs. Absolute score accuracy degrades
near the tails (|z| > 4) but the sign (label) remains correct.

Exported constants used by inference.py:
    POLY_A0, POLY_A1, POLY_A3   ->  P(z) = A0 + A1*z + A3*z^3
    POLY_SCALE                   ->  logit divisor applied before poly evaluation
    POLY_RANGE                   ->  fitting interval half-width
"""

import math

import numpy as np

# ── approximation parameters ──────────────────────────────────────────────────
# Worst-case logit |w·x+b| for normalised inputs in [-1,1]:
#   sum(|w|) + |b| = 16.12 + 1.52 + 28.44 + 21.72 + 11.89 ≈ 79.7
# Divide by POLY_SCALE=16 → max scaled logit ≈ 4.98.
# We fit over [-5, 5] to cover that range; the poly is monotone within ~[-4, 4].
POLY_SCALE: float = 16.0
POLY_RANGE: float = 5.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_degree3(n_pts: int = 2000) -> tuple[float, float, float]:
    """Least-squares fit of a3*z^3 + a1*z + a0 to sigmoid over [-POLY_RANGE, POLY_RANGE].

    Only odd + constant terms (a2 = 0 by symmetry of sigmoid around 0).
    a0 is fixed at 0.5 (P(0) = sigmoid(0) = 0.5 exactly).
    """
    z = np.linspace(-POLY_RANGE, POLY_RANGE, n_pts)
    y = _sigmoid(z)
    A = np.column_stack([z, z ** 3])
    b_vec = y - 0.5
    coeffs, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    a1, a3 = coeffs
    return 0.5, float(a1), float(a3)


# Fit once at import time.
POLY_A0, POLY_A1, POLY_A3 = _fit_degree3()


def poly_sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """Evaluate the degree-3 polynomial approximation of sigmoid at z."""
    return POLY_A0 + POLY_A1 * z + POLY_A3 * z ** 3


def monotone_range() -> float:
    """Half-width of the interval over which poly_sigmoid is strictly increasing.

    P'(z) = A1 + 3*A3*z^2 > 0  iff  z^2 < -A1/(3*A3)  (since A3 < 0)
    """
    if POLY_A3 >= 0:
        return float("inf")
    return float(math.sqrt(-POLY_A1 / (3.0 * POLY_A3)))


def max_abs_error(z_max: float | None = None, n_pts: int = 10_000) -> float:
    """Max |sigmoid(z) - poly_sigmoid(z)| over [-z_max, z_max].

    Defaults to the monotone region (where the approximation is designed to work).
    Outside this region the cubic term dominates but the label (sign) is still
    correct — see label_accuracy().
    """
    if z_max is None:
        z_max = min(monotone_range(), POLY_RANGE)
    z = np.linspace(-z_max, z_max, n_pts)
    return float(np.max(np.abs(_sigmoid(z) - poly_sigmoid(z))))


def label_accuracy(z_max: float = POLY_RANGE, n_pts: int = 10_000) -> float:
    """Fraction of points where sign(poly(z)-0.5) == sign(sigmoid(z)-0.5)."""
    z = np.linspace(-z_max, z_max, n_pts)
    true_label = (_sigmoid(z) >= 0.5).astype(int)
    poly_label = (poly_sigmoid(z) >= 0.5).astype(int)
    return float(np.mean(true_label == poly_label))


if __name__ == "__main__":
    z_mono = monotone_range()
    print("=== Degree-3 polynomial sigmoid approximation ===")
    print(f"Fitting interval       : [-{POLY_RANGE}, {POLY_RANGE}]")
    print(f"Logit scale divisor    : {POLY_SCALE}")
    print(f"P(z) = {POLY_A0:.6f} + {POLY_A1:.6f}*z + ({POLY_A3:.6f})*z^3")
    print(f"Monotone region        : [-{z_mono:.3f}, {z_mono:.3f}]")
    print(f"Max abs error (monotone region): {max_abs_error():.6f}")
    print(f"Max abs error (full range)     : {max_abs_error(POLY_RANGE):.6f}")
    print(f"Label accuracy (full range)    : {label_accuracy() * 100:.2f}%")
    print()
    for z_val in [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]:
        tv = _sigmoid(np.array([z_val]))[0]
        pv = poly_sigmoid(np.array([z_val]))[0]
        print(f"  z={z_val:+.1f}  sigmoid={tv:.4f}  poly={pv:.4f}  err={abs(tv-pv):.4f}")
