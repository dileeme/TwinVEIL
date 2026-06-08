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

    # Scale weights and bias into the poly sigmoid's valid range [-5, 5].
    # Worst-case |w·x+b| <= sum(|w|)+|b| ≈ 79.7; dividing by 16 keeps it <= 4.98.
    # Scaling in numpy (not on the ciphertext) keeps the CKKS scale at 2^40.
    # The label threshold (score >= 0.5) is preserved because sigmoid is monotone:
    # sigmoid(w·x+b) >= 0.5  iff  w·x+b >= 0  iff  (w/16)·x + b/16 >= 0.
    _S = 16.0
    w_s = (w / _S).tolist()
    b_s = b / _S

    dot_enc    = x_enc.dot(w_s)
    linear_enc = dot_enc + b_s

    # Degree-3 polynomial sigmoid: σ(t) ≈ 0.5 + 0.197t − 0.004t³
    sq_enc    = linear_enc * linear_enc   # level +1
    cubic_enc = sq_enc * linear_enc       # level +2
    score_enc = linear_enc * 0.197 - cubic_enc * 0.004 + 0.5

    return base64.b64encode(score_enc.serialize()).decode()
