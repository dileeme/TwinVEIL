"""
client/encryptor.py — CKKS encryption for vehicle telemetry (Module 1).

Replaces the plaintext normalise() path with encrypt():
  raw readings → normalise → CKKS ciphertext → serialised bytes for HTTP.

The secret key never leaves this module. The public evaluation context
(no secret key) is exposed for upload to the server.
"""

import json
import base64
from pathlib import Path

import numpy as np
import tenseal as ts

# ── canonical feature order (must match Module 2 contract) ──────────────────
FEATURE_ORDER = ["speed", "longitudinal_accel", "steering_angle", "brake_pressure"]

# ── normalisation bounds from models/scaler_bounds.json ────────────────────
_BOUNDS_PATH = Path(__file__).parent.parent / "models" / "scaler_bounds.json"

# ── CKKS context (created once; secret key stays here) ─────────────────────
_context: ts.Context | None = None


def _build_context() -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=2**14,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    return ctx


def get_context() -> ts.Context:
    """Return the full (secret-key) CKKS context, creating it on first call."""
    global _context
    if _context is None:
        _context = _build_context()
    return _context


def public_context_b64() -> str:
    """Serialise the evaluation context (no secret key) as a base-64 string."""
    ctx = get_context()
    pub_bytes = ctx.serialize(save_secret_key=False)
    return base64.b64encode(pub_bytes).decode()


# ── normalisation ────────────────────────────────────────────────────────────

def _load_bounds() -> dict:
    with open(_BOUNDS_PATH) as f:
        return json.load(f)


def normalise(raw: dict) -> np.ndarray:
    """
    Map raw sensor readings to [-1, 1] using the fixed scaler bounds.

    raw: dict with keys matching FEATURE_ORDER.
    Returns a float64 ndarray of shape (4,).
    """
    bounds = _load_bounds()
    out = []
    for feat in FEATURE_ORDER:
        lo, hi = bounds[feat]
        val = raw[feat]
        normed = np.clip(2.0 * (val - lo) / (hi - lo) - 1.0, -1.0, 1.0)
        out.append(float(normed))
    return np.array(out, dtype=np.float64)


# ── encryption ───────────────────────────────────────────────────────────────

def encrypt(raw: dict) -> str:
    """
    Normalise raw telemetry and encrypt into a CKKS ciphertext.

    Returns a base-64 encoded serialised CKKSVector (single slot per feature
    packed into one vector).
    """
    x_norm = normalise(raw)
    ctx = get_context()
    enc = ts.ckks_vector(ctx, x_norm.tolist())
    return base64.b64encode(enc.serialize()).decode()


# ── decryption ───────────────────────────────────────────────────────────────

def decrypt_result(encrypted_b64: str) -> float:
    """
    Decrypt a base-64 encoded CKKS scalar result returned by the server.

    Returns the first slot value (the anomaly score).
    """
    ctx = get_context()
    raw_bytes = base64.b64decode(encrypted_b64)
    enc_vec = ts.lazy_ckks_vector_from(raw_bytes)
    enc_vec.link_context(ctx)
    values = enc_vec.decrypt()
    return float(values[0])
