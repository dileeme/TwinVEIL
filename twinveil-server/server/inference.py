"""
TwinVEIL — Module 2, Step 4: inference logic (plaintext for now).

This is the ONLY file that changes when the CKKS encrypted pipeline is
connected (§4.1): predict_plaintext() will be joined/replaced by a
predict_encrypted() that runs the same w.x + b and degree-3 polynomial sigmoid
on ciphertext. The exported weights (lr_weights.npy / lr_bias.npy) are the
contract shared with the HE stage.
"""
from pathlib import Path
import numpy as np

# Anchor model paths to the twinveil-server root so the service runs from any
# working directory (the spec used bare 'models/...' relative paths).
_ROOT = Path(__file__).resolve().parent.parent
_w = np.load(_ROOT / "models" / "lr_weights.npy")        # shape (4,), float64
_b = float(np.load(_ROOT / "models" / "lr_bias.npy"))    # scalar


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def predict_plaintext(features: list) -> dict:
    x = np.array(features, dtype=np.float64)
    z = float(x @ _w + _b)
    score = _sigmoid(z)
    return {"anomaly_score": round(score, 6), "label": int(score >= 0.5)}
