"""
client/telemetry_client.py — sends encrypted telemetry to the HE server.

Flow:
  1. Upload the public evaluation context once (or on server restart).
  2. For each reading: encrypt locally → POST ciphertext → receive encrypted
     result → decrypt locally → return anomaly score + label.

The secret key never leaves this process.
"""

import argparse
import json
import sys
from typing import Optional

import requests

from encryptor import encrypt, decrypt_result, public_context_b64

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


# ── context upload ────────────────────────────────────────────────────────────

def upload_context(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Push the public evaluation context to the server."""
    pub_ctx = public_context_b64()
    resp = requests.post(
        f"{base_url}/upload_context",
        json={"pub_ctx": pub_ctx},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("status") == "ok"


# ── encrypted prediction ──────────────────────────────────────────────────────

def predict_encrypted(
    vehicle_id: str,
    raw: dict,
    base_url: str = DEFAULT_BASE_URL,
) -> dict:
    """
    Encrypt raw telemetry, send to server, decrypt result.

    Returns {"vehicle_id": str, "anomaly_score": float, "label": int}.
    """
    ciphertext_b64 = encrypt(raw)

    resp = requests.post(
        f"{base_url}/predict_he",
        json={"vehicle_id": vehicle_id, "ciphertext": ciphertext_b64},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()

    score = decrypt_result(body["encrypted_result"])
    label = int(score >= 0.5)

    return {
        "vehicle_id": vehicle_id,
        "anomaly_score": round(score, 6),
        "label": label,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TwinVEIL encrypted telemetry client")
    p.add_argument("--url", default=DEFAULT_BASE_URL, help="Server base URL")
    p.add_argument("--vehicle-id", default="v001")
    p.add_argument("--speed", type=float, required=True)
    p.add_argument("--accel", type=float, required=True, dest="longitudinal_accel")
    p.add_argument("--steering", type=float, required=True, dest="steering_angle")
    p.add_argument("--brake", type=float, required=True, dest="brake_pressure")
    p.add_argument(
        "--upload-ctx", action="store_true",
        help="Upload public context before sending prediction",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    raw = {
        "speed": args.speed,
        "longitudinal_accel": args.longitudinal_accel,
        "steering_angle": args.steering_angle,
        "brake_pressure": args.brake_pressure,
    }

    if args.upload_ctx:
        print("Uploading public evaluation context ...", flush=True)
        upload_context(args.url)
        print("Context uploaded.", flush=True)

    result = predict_encrypted(args.vehicle_id, raw, args.url)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
