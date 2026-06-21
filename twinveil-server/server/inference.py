"""
twinveil-server/server/inference.py — plaintext + HE inference.

Plaintext path (Module 2, unchanged):
    predict_plaintext(features) → {"anomaly_score": float, "label": int}

Homomorphic path (Module 1):
    predict_encrypted(ciphertext_b64, pub_ctx_b64) → encrypted_result_b64

    The degree-3 poly sigmoid approximation is:
        σ(x) ≈ 0.5 + 0.197x − 0.004x³
    valid over ≈ [-5, 5], which covers w·x+b on normalised inputs.

The server NEVER holds the secret key.
"""

import base64
from functools import lru_cache
from pathlib import Path

import numpy as np
import tenseal as ts

from .poly_approx import POLY_A0, POLY_A1, POLY_A3, POLY_SCALE

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"


# ── weight loading ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_weights() -> tuple[np.ndarray, float]:
    w = np.load(_MODELS_DIR / "lr_weights.npy")   # shape (4,)
    b = float(np.load(_MODELS_DIR / "lr_bias.npy"))
    return w, b


# ── plaintext inference (Module 2 contract — do not change signature) ─────────

def predict_plaintext(features: list[float]) -> dict:
    """
    features: list of 4 floats, already normalised to [-1, 1].
    Returns {"anomaly_score": float, "label": 0|1}.
    """
    w, b = _load_weights()
    x = np.array(features, dtype=np.float64)
    logit = float(np.dot(w, x) + b)
    score = _sigmoid(logit)
    return {"anomaly_score": round(score, 6), "label": int(score >= 0.5)}


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


# ── homomorphic inference (Module 1) ─────────────────────────────────────────

def predict_encrypted(ciphertext_b64: str, pub_ctx_b64: str) -> str:
    """
    Perform w·x_enc + b homomorphically, then apply poly sigmoid.

    ciphertext_b64 : base-64 serialised CKKSVector of the 4 normalised features.
    pub_ctx_b64    : base-64 serialised public evaluation context (no secret key).

    Returns base-64 serialised CKKSVector containing the encrypted anomaly score.
    """
    w, b = _load_weights()

    # Deserialise the public context (evaluation only, no secret key)
    pub_ctx_bytes = base64.b64decode(pub_ctx_b64)
    ctx = ts.context_from(pub_ctx_bytes)

    # Deserialise the ciphertext
    ct_bytes = base64.b64decode(ciphertext_b64)
    x_enc = ts.lazy_ckks_vector_from(ct_bytes)
    x_enc.link_context(ctx)

    # Scale weights into poly_approx's fitted range [-POLY_RANGE, POLY_RANGE].
    # Worst-case |w·x+b| ≈ 79.7 → dividing by POLY_SCALE keeps it within range.
    # Sign is preserved (sigmoid monotone), so label (score >= 0.5) is unchanged.
    w_s = (w / POLY_SCALE).tolist()
    b_s = b / POLY_SCALE

    dot_enc    = x_enc.dot(w_s)
    linear_enc = dot_enc + b_s

    # Degree-3 polynomial sigmoid (coefficients from poly_approx.py):
    #   P(t) = POLY_A0 + POLY_A1*t + POLY_A3*t^3
    sq_enc    = linear_enc * linear_enc          # depth +1
    cubic_enc = sq_enc * linear_enc              # depth +2
    score_enc = linear_enc * POLY_A1 + cubic_enc * POLY_A3 + POLY_A0

    return base64.b64encode(score_enc.serialize()).decode()
