"""
TwinVEIL — Module 2, Step 4.7: endpoint test.

Drives the real ASGI app (server.main) in-process with FastAPI's TestClient,
exercising the same code path a live `uvicorn server.main:app` would:
  POST /predict (normal sample)  -> expect label 0
  POST /predict (anomaly sample) -> expect label 1
  GET  /state/v001               -> last stored twin state
  GET  /health                   -> active Redis backend

Writes reports/step4_test.txt. The equivalent curl commands (per §4.7) are
included in the report for manual use against a live server.

Run as a script:  python step4_test.py
Run via pytest:   pytest step4_test.py
"""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server.main import app
from server.state_store import BACKEND

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "step4_test.txt"
FEATURES = ["speed_norm", "longitudinal_accel_norm",
            "steering_angle_norm", "brake_pressure_norm"]

# Partner-Procedure illustrative vectors (reference only).
DOC_NORMAL = {"vehicle_id": "vdoc", "features": [0.1, 0.05, -0.02, 0.0]}
DOC_ANOMALY = {"vehicle_id": "vdoc", "features": [0.85, -0.90, 0.0, 0.95]}


def _load_samples() -> tuple[dict, dict]:
    """Pull one real normal and one real anomaly row from labelled.csv."""
    df = pd.read_csv(ROOT / "data" / "processed" / "labelled.csv")
    norm_row = df[df["label"] == 0].iloc[0][FEATURES].round(4).tolist()
    anom_row = df[df["label"] == 1].iloc[0][FEATURES].round(4).tolist()
    return (
        {"vehicle_id": "v001", "features": norm_row},
        {"vehicle_id": "v001", "features": anom_row},
    )


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def samples():
    return _load_samples()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok"
    assert "redis_backend" in r


def test_predict_normal(client, samples):
    normal, _ = samples
    r = client.post("/predict", json=normal).json()
    assert r["label"] == 0, f"expected label=0, got {r}"


def test_predict_anomaly(client, samples):
    _, anomaly = samples
    r = client.post("/predict", json=anomaly).json()
    assert r["label"] == 1, f"expected label=1, got {r}"


def test_state_roundtrip(client, samples):
    _, anomaly = samples
    last = client.post("/predict", json=anomaly).json()
    state = client.get("/state/v001").json()
    assert state == last, "stored twin state should match last prediction"


def test_doc_anomaly(client):
    r = client.post("/predict", json=DOC_ANOMALY).json()
    assert r["label"] == 1, f"doc anomaly should be label=1, got {r}"


def test_invalid_feature_length(client):
    """Wrong-length feature vectors must return 422, not 500."""
    r = client.post("/predict", json={"vehicle_id": "v_bad", "features": [0.1, 0.2]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Script entrypoint — runs all tests and writes the report
# ---------------------------------------------------------------------------

def _run_and_report():
    normal, anomaly = _load_samples()
    c = TestClient(app)

    health = c.get("/health").json()
    r_normal = c.post("/predict", json=normal).json()
    r_anomaly = c.post("/predict", json=anomaly).json()
    r_state = c.get("/state/v001").json()
    r_doc_normal = c.post("/predict", json=DOC_NORMAL).json()
    r_doc_anomaly = c.post("/predict", json=DOC_ANOMALY).json()

    lines = [
        "=== Step 4.7 Service Test (TestClient against the real ASGI app) ===",
        f"Redis backend: {BACKEND}",
        f"/health -> {json.dumps(health)}",
        "",
        "# Real NORMAL row from labelled.csv (expect label=0)",
        f"  features={json.dumps(normal['features'])}",
        f"-> {json.dumps(r_normal)}",
        "",
        "# Real ANOMALY row from labelled.csv (expect label=1)",
        f"  features={json.dumps(anomaly['features'])}",
        f"-> {json.dumps(r_anomaly)}",
        "",
        "# Twin state for v001 (should equal the last /predict result)",
        "curl http://localhost:8000/state/v001",
        f"-> {json.dumps(r_state)}",
        "",
        "--- Partner-Procedure illustrative vectors (reference) ---",
        "# doc 'normal' [0.1,0.05,-0.02,0.0]: brake_norm=0.0 == brake 0.5 (mid-brake,",
        "#   off the normal distribution) -> the model reasonably flags it.",
        f"-> {json.dumps(r_doc_normal)}",
        "# doc 'anomaly' [0.85,-0.90,0.0,0.95]: high speed + hard brake",
        f"-> {json.dumps(r_doc_anomaly)}",
        "",
        "=== Assertions ===",
        f"real normal  label == 0 : {r_normal['label'] == 0}",
        f"real anomaly label == 1 : {r_anomaly['label'] == 1}",
        f"doc  anomaly label == 1 : {r_doc_anomaly['label'] == 1}",
        f"state round-trips        : {r_state == r_anomaly}",
    ]
    report = "\n".join(lines)
    REPORT.write_text(report + "\n", encoding="utf-8")
    print(report)

    assert r_normal["label"] == 0, "real normal sample should be label 0"
    assert r_anomaly["label"] == 1, "real anomaly sample should be label 1"
    assert r_doc_anomaly["label"] == 1, "doc anomaly sample should be label 1"
    assert r_state == r_anomaly, "stored twin state should match last prediction"
    print("\nAll Step 4 assertions passed.")


if __name__ == "__main__":
    _run_and_report()
