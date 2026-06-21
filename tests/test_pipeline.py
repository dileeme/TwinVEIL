"""
tests/test_pipeline.py — end-to-end Module 1 encrypted pipeline tests.

Runs entirely in-process (no live HTTP server required) using FastAPI's
TestClient. Validates the complete flow:

  1. CKKS context created in trusted client domain
  2. Public context (no secret key) uploaded to server
  3. Feature vector encrypted on client
  4. Server runs encrypted w·x+b + degree-3 poly sigmoid on ciphertext
  5. Server NEVER decrypts — returns encrypted result
  6. Client decrypts and interprets anomaly score + label

Tests also verify:
  - Poly sigmoid coefficients and label accuracy
  - HE label matches plaintext label on held-out samples
  - Ciphertext never contains plaintext values (semantic security smoke-test)
  - /predict (plaintext Module 2 path) still works alongside /predict_he
  - 422 on missing context, bad feature length

Run:  pytest tests/test_pipeline.py -v
"""
import base64
import sys
from pathlib import Path

import numpy as np
import pytest
import tenseal as ts
from fastapi.testclient import TestClient

# ── path setup (allows running from project root or tests/) ───────────────────
ROOT = Path(__file__).resolve().parent.parent
# client/ → allows `from encryptor import ...`
sys.path.insert(0, str(ROOT / "client"))
# twinveil-server/ → allows `from server.main import app` etc.
sys.path.insert(0, str(ROOT / "twinveil-server"))

from server.main import app                                     # noqa: E402
from server.poly_approx import (                                # noqa: E402
    POLY_A0, POLY_A1, POLY_A3, POLY_SCALE, POLY_RANGE,
    poly_sigmoid, max_abs_error, label_accuracy, monotone_range,
)
from encryptor import (                              # noqa: E402
    get_context, encrypt, decrypt_result, public_context_b64, normalise,
)

# ── test fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def ctx_uploaded(client):
    """Upload public CKKS context to the server once per test module."""
    pub_ctx = public_context_b64()
    resp = client.post("/upload_context", json={"pub_ctx": pub_ctx})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    return pub_ctx


# ── sample telemetry ──────────────────────────────────────────────────────────

NORMAL_RAW = {
    "speed": 25.0,
    "longitudinal_accel": 0.3,
    "steering_angle": 5.0,
    "brake_pressure": 0.02,
}
ANOMALY_RAW = {
    "speed": 30.0,              # fast
    "longitudinal_accel": -5.0, # hard brake
    "steering_angle": 8.0,
    "brake_pressure": 0.90,     # high brake
}
ANOMALY_STEER_RAW = {           # Rule 2: steering oscillation
    "speed": 20.0,
    "longitudinal_accel": 0.1,
    "steering_angle": 350.0,    # |steering| > 200
    "brake_pressure": 0.01,
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Polynomial sigmoid tests
# ═════════════════════════════════════════════════════════════════════════════

def test_poly_sigmoid_at_zero():
    """P(0) must equal exactly 0.5."""
    assert POLY_A0 == pytest.approx(0.5, abs=1e-9)
    assert poly_sigmoid(0.0) == pytest.approx(0.5, abs=1e-9)


def test_poly_sigmoid_monotone():
    """Poly must be strictly increasing within its monotone region.

    A degree-3 poly with A3 < 0 has inflection points at ±z* where
    P'(z*) = 0. It is monotone on (-z*, z*) and non-monotone outside.
    We check monotonicity within 95% of z* to avoid floating-point edge effects.
    """
    z_star = monotone_range()
    check_range = 0.95 * z_star          # safely inside the monotone region
    z = np.linspace(-check_range, check_range, 500)
    vals = poly_sigmoid(z)
    assert np.all(np.diff(vals) > 0), \
        f"poly_sigmoid is not monotone on [-{check_range:.2f}, {check_range:.2f}]"


def test_poly_sigmoid_max_error():
    """Max absolute error within the monotone region must be below 5%.

    The degree-3 poly is an approximation: within its monotone region
    (|z| < z* ≈ 4), abs error stays below 5%. Outside this region the
    cubic dominates and absolute error grows, but the label (sign) stays
    correct — verified separately in test_poly_sigmoid_label_accuracy.
    """
    err = max_abs_error()   # defaults to the monotone region
    assert err < 0.05, f"poly sigmoid max error in monotone region {err:.4f} exceeds 5%"


def test_poly_sigmoid_label_accuracy():
    """Sign agreement with sigmoid (i.e. label accuracy) must be ≥ 99%."""
    acc = label_accuracy()
    assert acc >= 0.99, f"label accuracy {acc:.4f} < 99%"


# ═════════════════════════════════════════════════════════════════════════════
# 2. Encryption / decryption round-trip
# ═════════════════════════════════════════════════════════════════════════════

def test_encrypt_decrypt_roundtrip():
    """Decrypt(encrypt(x)) must recover x to within CKKS noise tolerance."""
    raw = NORMAL_RAW
    x_plain = normalise(raw)

    ctx = get_context()
    enc = ts.ckks_vector(ctx, x_plain.tolist())
    recovered = np.array(enc.decrypt())

    np.testing.assert_allclose(recovered, x_plain, atol=1e-4,
                               err_msg="CKKS round-trip error exceeds 1e-4")


def test_ciphertext_hides_plaintext():
    """
    Serialised ciphertext bytes must not literally contain the ASCII
    representation of the plaintext values — a basic sanity check that
    the bytes look like ciphertext rather than plaintext JSON.
    """
    ct_b64 = encrypt(NORMAL_RAW)
    ct_bytes = base64.b64decode(ct_b64)
    x_plain = normalise(NORMAL_RAW)
    for val in x_plain:
        # Convert to bytes and check it doesn't appear verbatim in the ciphertext
        val_bytes = str(round(val, 4)).encode()
        assert val_bytes not in ct_bytes, (
            f"plaintext value {val} found literally in ciphertext bytes"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. HTTP endpoint tests — plaintext path (Module 2, regression guard)
# ═════════════════════════════════════════════════════════════════════════════

def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok"
    assert "redis" in r


def test_predict_plaintext_normal(client):
    norm = normalise(NORMAL_RAW).tolist()
    r = client.post("/predict", json={"vehicle_id": "v_pt_normal",
                                      "features": norm}).json()
    assert r["label"] == 0, f"plaintext normal: expected label=0, got {r}"


def test_predict_plaintext_anomaly(client):
    norm = normalise(ANOMALY_RAW).tolist()
    r = client.post("/predict", json={"vehicle_id": "v_pt_anom",
                                      "features": norm}).json()
    assert r["label"] == 1, f"plaintext anomaly: expected label=1, got {r}"


def test_predict_plaintext_bad_length(client):
    r = client.post("/predict", json={"vehicle_id": "v_bad",
                                      "features": [0.1, 0.2]})
    assert r.status_code == 422


# ═════════════════════════════════════════════════════════════════════════════
# 4. HTTP endpoint tests — encrypted path (Module 1, the real thing)
# ═════════════════════════════════════════════════════════════════════════════

def test_predict_he_requires_context(client):
    """Calling /predict_he before uploading context must return 400."""
    # Use a fresh TestClient so context store is empty
    fresh = TestClient(app)
    ct = encrypt(NORMAL_RAW)
    r = fresh.post("/predict_he", json={"vehicle_id": "v_noctx",
                                        "ciphertext": ct})
    assert r.status_code == 400
    assert "context" in r.json()["detail"].lower()


def test_upload_context(client, ctx_uploaded):
    """Context upload must succeed and return status ok."""
    # ctx_uploaded fixture already asserted this; just confirm it ran
    assert ctx_uploaded is not None


def test_predict_he_normal(client, ctx_uploaded):
    """Encrypted inference on a normal sample must yield label=0."""
    ct = encrypt(NORMAL_RAW)
    r = client.post("/predict_he",
                    json={"vehicle_id": "v_he_normal", "ciphertext": ct})
    assert r.status_code == 200, r.text
    body = r.json()
    score = decrypt_result(body["encrypted_result"])
    label = int(score >= 0.5)
    assert label == 0, f"HE normal: score={score:.4f} expected label=0"


def test_predict_he_anomaly_hard_brake(client, ctx_uploaded):
    """Encrypted inference on a hard-brake anomaly must yield label=1."""
    ct = encrypt(ANOMALY_RAW)
    r = client.post("/predict_he",
                    json={"vehicle_id": "v_he_anom", "ciphertext": ct})
    assert r.status_code == 200, r.text
    score = decrypt_result(r.json()["encrypted_result"])
    assert int(score >= 0.5) == 1, f"HE anomaly: score={score:.4f} expected label=1"


def test_predict_he_anomaly_steering(client, ctx_uploaded):
    """Encrypted inference on a steering-oscillation anomaly must yield label=1."""
    ct = encrypt(ANOMALY_STEER_RAW)
    r = client.post("/predict_he",
                    json={"vehicle_id": "v_he_steer", "ciphertext": ct})
    assert r.status_code == 200, r.text
    score = decrypt_result(r.json()["encrypted_result"])
    assert int(score >= 0.5) == 1, f"HE steering anomaly: score={score:.4f} expected label=1"


def test_he_label_matches_plaintext(client, ctx_uploaded):
    """
    HE label must agree with the plaintext label on every test sample.
    This is the core privacy-correctness guarantee: the encrypted computation
    must produce the same decision as the unencrypted one.
    """
    samples = [
        ("normal",         NORMAL_RAW,       0),
        ("hard_brake",     ANOMALY_RAW,      1),
        ("steer_osc",      ANOMALY_STEER_RAW,1),
        ("mild_cruise",    {"speed": 20.0, "longitudinal_accel": 0.5,
                            "steering_angle": 2.0, "brake_pressure": 0.0}, 0),
    ]
    for name, raw, expected_label in samples:
        # Plaintext label
        norm = normalise(raw).tolist()
        pt_r = client.post("/predict", json={"vehicle_id": f"v_{name}",
                                             "features": norm}).json()
        pt_label = pt_r["label"]

        # HE label
        ct = encrypt(raw)
        he_r = client.post("/predict_he",
                           json={"vehicle_id": f"v_{name}_he",
                                 "ciphertext": ct}).json()
        he_score = decrypt_result(he_r["encrypted_result"])
        he_label = int(he_score >= 0.5)

        assert pt_label == expected_label, \
            f"{name}: plaintext label={pt_label} != expected {expected_label}"
        assert he_label == pt_label, \
            (f"{name}: HE label={he_label} (score={he_score:.4f}) "
             f"!= plaintext label={pt_label}")


def test_result_is_encrypted(client, ctx_uploaded):
    """
    The server must return bytes that are NOT the plaintext score.
    Decrypt the result and confirm the raw bytes don't reveal the score value.
    """
    ct = encrypt(NORMAL_RAW)
    r = client.post("/predict_he",
                    json={"vehicle_id": "v_enc_check", "ciphertext": ct})
    body = r.json()
    result_bytes = base64.b64decode(body["encrypted_result"])
    score = decrypt_result(body["encrypted_result"])
    score_bytes = str(round(score, 4)).encode()
    assert score_bytes not in result_bytes, \
        "Encrypted result appears to contain the plaintext score — not encrypted!"
