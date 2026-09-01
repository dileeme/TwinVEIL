"""
tests/test_model_contract.py — enforces the Module 2 → Module 1 model contract.

Module 1 does not train anything. Its weights are produced by step3_train.py on
the Twinveil-module2 branch and copied in verbatim. Nothing used to check that,
and in September 2026 the copies silently diverged: Module 1 was carrying a
4-decimal transcription of the weights (max delta ~5e-5) while Module 2 reported
metrics for the full-precision export. The encrypted path was therefore serving a
different model than the one the paper measured.

These tests close that hole. They enforce, against models/MANIFEST.json:

  1. Byte-level integrity  — sha256 of every artefact
  2. Structural contract   — shape, dtype and feature count
  3. HE compatibility      — the trained weights must stay inside the range where
                             the degree-3 polynomial activation preserves labels

Test 3 is the important one. The encrypted path divides the logit by POLY_SCALE
before evaluating P(t) = A0 + A1*t + A3*t^3. Because A3 < 0, P(t) rises above 0.5,
peaks, and then falls back through 0.5 at t = sqrt(-A1/A3). Any retraining that
pushes sum|w| + |b| past POLY_SCALE * that root makes the most anomalous possible
input decrypt to label 0 — silently, with every existing test still green.

Run:  pytest tests/test_model_contract.py -v
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "twinveil-server"))

from server.poly_approx import (  # noqa: E402
    POLY_A0, POLY_A1, POLY_A3, POLY_SCALE, poly_sigmoid,
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(MODELS / "MANIFEST.json") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── 1. byte-level integrity ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "filename", ["lr_weights.npy", "lr_bias.npy", "scaler_bounds.json"]
)
def test_artefact_checksum(manifest, filename):
    """Each artefact must match the checksum recorded when it was imported.

    A failure here means the file was regenerated, hand-edited or retyped. Do not
    update the manifest to make this pass unless you have deliberately re-imported
    from Twinveil-module2 — copy the file verbatim rather than reconstructing it.
    """
    expected = manifest["artefacts"][filename]["sha256"]
    actual = _sha256(MODELS / filename)
    assert actual == expected, (
        f"{filename} has drifted from the Module 2 export.\n"
        f"  expected {expected}\n"
        f"  actual   {actual}\n"
        f"  source   {manifest['source']['branch']} @ {manifest['source']['commit'][:12]}"
    )


# ── 2. structural contract ───────────────────────────────────────────────────

def test_weights_shape_and_dtype(manifest):
    """Weights must stay (4,) float64 — the fixed 4-feature contract."""
    spec = manifest["artefacts"]["lr_weights.npy"]
    w = np.load(MODELS / "lr_weights.npy")
    assert w.shape == tuple(spec["shape"]), f"expected shape {spec['shape']}, got {w.shape}"
    assert w.dtype == np.dtype(spec["dtype"]), f"expected {spec['dtype']}, got {w.dtype}"
    assert len(spec["feature_order"]) == w.shape[0]


def test_bias_is_float64_scalar(manifest):
    b = np.load(MODELS / "lr_bias.npy")
    assert b.shape == (), f"bias must be a scalar, got shape {b.shape}"
    assert b.dtype == np.float64


def test_scaler_bounds_cover_every_feature(manifest):
    """Every feature the model consumes must have a normalisation bound."""
    with open(MODELS / "scaler_bounds.json") as f:
        bounds = json.load(f)
    for feat in manifest["artefacts"]["lr_weights.npy"]["feature_order"]:
        assert feat in bounds, f"scaler_bounds.json is missing {feat!r}"
        lo, hi = bounds[feat]
        assert lo < hi, f"{feat}: degenerate bounds [{lo}, {hi}]"


# ── 3. HE compatibility — the guard that actually matters ────────────────────

def _poly_positive_root() -> float:
    """Largest t > 0 with P(t) = 0.5, i.e. where the cubic drags the curve back down.

    P(t) - 0.5 = A1*t + A3*t^3 = t*(A1 + A3*t^2), so the non-zero roots are
    t = +/- sqrt(-A1/A3), real because A3 < 0 < A1.
    """
    assert POLY_A3 < 0 < POLY_A1, "polynomial no longer has the expected shape"
    return float(np.sqrt(-POLY_A1 / POLY_A3))


def test_manifest_he_constants_match_the_polynomial(manifest):
    """The manifest's recorded HE constants must match the live polynomial."""
    he = manifest["he_compatibility"]
    assert POLY_SCALE == pytest.approx(he["poly_scale"])
    assert _poly_positive_root() == pytest.approx(he["poly_positive_root"], rel=1e-9)
    assert POLY_A0 == pytest.approx(0.5, abs=1e-12), "P(0) must be exactly 0.5"


def test_worst_case_logit_stays_in_the_label_preserving_range(manifest):
    """The trained weights must keep the scaled logit inside P(t) > 0.5.

    Worst case over normalised inputs x in [-1,1]^4 is sum|w| + |b|, attained at
    x = sign(w). If (sum|w| + |b|) / POLY_SCALE >= the positive root, that input
    decrypts to label 0 despite being maximally anomalous.
    """
    w = np.load(MODELS / "lr_weights.npy")
    b = float(np.load(MODELS / "lr_bias.npy"))
    worst = float(np.abs(w).sum() + abs(b))
    scaled = worst / POLY_SCALE
    root = _poly_positive_root()

    assert scaled < root, (
        f"Model is incompatible with the polynomial activation.\n"
        f"  worst-case |w.x+b|  = {worst:.4f}\n"
        f"  scaled (/{POLY_SCALE:g})       = {scaled:.4f}\n"
        f"  label-preserving max = {root:.4f}\n"
        f"Raise POLY_SCALE to > {worst / root:.2f}, refit the polynomial over the "
        f"real logit range, or retrain with stronger regularisation."
    )

    # Recorded margin must not have silently shrunk either.
    assert (root - scaled) == pytest.approx(manifest["he_compatibility"]["margin"], rel=1e-6)


def test_extreme_input_still_labels_as_anomalous():
    """End-to-end sanity: x = sign(w) is maximally anomalous and must label 1."""
    w = np.load(MODELS / "lr_weights.npy")
    b = float(np.load(MODELS / "lr_bias.npy"))
    x = np.sign(w)
    z = float(w @ x + b)
    assert z > 0, "sign(w) should produce a positive logit"
    assert poly_sigmoid(z / POLY_SCALE) >= 0.5, (
        f"maximally anomalous input decrypts to label 0 "
        f"(logit {z:.2f}, scaled {z / POLY_SCALE:.3f}) — the cubic term has taken over"
    )
