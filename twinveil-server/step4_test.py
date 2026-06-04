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
"""
import json
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
from server.main import app
from server.state_store import BACKEND

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "step4_test.txt"
FEATURES = ["speed_norm", "longitudinal_accel_norm",
            "steering_angle_norm", "brake_pressure_norm"]

# Pull a genuine normal row and a genuine anomaly row from the labelled data so
# the test reflects this project's actual normalisation/distribution.
df = pd.read_csv(ROOT / "data" / "processed" / "labelled.csv")
norm_row = df[df["label"] == 0].iloc[0][FEATURES].round(4).tolist()
anom_row = df[df["label"] == 1].iloc[0][FEATURES].round(4).tolist()

NORMAL = {"vehicle_id": "v001", "features": norm_row}
ANOMALY = {"vehicle_id": "v001", "features": anom_row}
# The Partner Procedure's illustrative vectors (note: its "normal" sets
# brake_norm=0.0 == brake 0.5, mid-braking, which is off the normal-driving
# distribution under the fixed [-1,1] bounds; shown here for reference).
DOC_NORMAL = {"vehicle_id": "vdoc", "features": [0.1, 0.05, -0.02, 0.0]}
DOC_ANOMALY = {"vehicle_id": "vdoc", "features": [0.85, -0.90, 0.0, 0.95]}

client = TestClient(app)

health = client.get("/health").json()
r_normal = client.post("/predict", json=NORMAL).json()
r_anomaly = client.post("/predict", json=ANOMALY).json()
r_state = client.get("/state/v001").json()
r_doc_normal = client.post("/predict", json=DOC_NORMAL).json()
r_doc_anomaly = client.post("/predict", json=DOC_ANOMALY).json()

lines = [
    "=== Step 4.7 Service Test (TestClient against the real ASGI app) ===",
    f"Redis backend: {BACKEND}",
    f"/health -> {json.dumps(health)}",
    "",
    "# Real NORMAL row from labelled.csv (expect label=0)",
    f"  features={json.dumps(NORMAL['features'])}",
    f"-> {json.dumps(r_normal)}",
    "",
    "# Real ANOMALY row from labelled.csv (expect label=1)",
    f"  features={json.dumps(ANOMALY['features'])}",
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
