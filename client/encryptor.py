"""
TwinVEIL — Module 2, client-side normaliser (plaintext baseline).

Accepts raw vehicle sensor readings, normalises them to [-1, 1] using the
fixed domain bounds from models/scaler_bounds.json, then POSTs to the
TwinVEIL inference service.

This is the plaintext stub. When the CKKS pipeline (Module 3) is connected,
normalise() is replaced by encrypt() and the payload becomes serialised
ciphertext bytes instead of a list of floats — only this file changes on the
client side, mirroring how only inference.py changes on the server side.
"""
import json
import sys
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
_BOUNDS_PATH = _ROOT / "models" / "scaler_bounds.json"
_DEFAULT_URL = "http://localhost:8000"

# Canonical feature order — must match the server's FEATURES list.
FEATURE_ORDER = ["speed", "longitudinal_accel", "steering_angle", "brake_pressure"]


def _load_bounds() -> dict:
    with open(_BOUNDS_PATH) as f:
        return json.load(f)


def normalise(raw: dict, bounds: dict | None = None) -> list[float]:
    """Map raw sensor readings to [-1, 1] using fixed domain bounds.

    Args:
        raw: dict with keys matching FEATURE_ORDER and float sensor values.
        bounds: optional pre-loaded bounds dict; loaded from disk if omitted.

    Returns:
        List of 4 floats in [-1, 1], one per feature in FEATURE_ORDER.
    """
    if bounds is None:
        bounds = _load_bounds()
    out = []
    for col in FEATURE_ORDER:
        lo, hi = bounds[col]
        v = float(raw[col])
        norm = 2.0 * (v - lo) / (hi - lo) - 1.0
        out.append(max(-1.0, min(1.0, norm)))  # clamp to [-1, 1]
    return out


def predict(vehicle_id: str, raw: dict, server_url: str = _DEFAULT_URL) -> dict:
    """Normalise raw sensor readings and POST to the inference service.

    Args:
        vehicle_id: identifier stored as the digital-twin key.
        raw: dict of raw sensor values (speed m/s, accel m/s^2, etc.).
        server_url: base URL of the TwinVEIL FastAPI server.

    Returns:
        Server response dict with keys ``anomaly_score`` and ``label``.
    """
    features = normalise(raw)
    payload = {"vehicle_id": vehicle_id, "features": features}
    resp = requests.post(f"{server_url}/predict", json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # Smoke-test against a running server (default: localhost:8000).
    # Usage: python client/encryptor.py [http://host:port]
    url = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_URL

    samples = [
        (
            "v_smoke_normal",
            {"speed": 25.0, "longitudinal_accel": 0.3,
             "steering_angle": 5.0, "brake_pressure": 0.05},
        ),
        (
            "v_smoke_anomaly",
            {"speed": 30.0, "longitudinal_accel": -5.0,
             "steering_angle": 8.0, "brake_pressure": 0.9},
        ),
    ]
    for vid, raw in samples:
        normed = normalise(raw)
        result = predict(vid, raw, server_url=url)
        print(f"{vid}: normed={[round(x, 4) for x in normed]}  result={result}")
