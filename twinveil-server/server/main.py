"""
TwinVEIL — Module 2, Step 4.6: FastAPI inference service.

Endpoints:
  POST /predict            -> run plaintext LR inference, persist twin state
  GET  /state/{vehicle_id} -> read back the last stored twin state
  GET  /health             -> liveness + which Redis backend is active
"""
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field

from server.inference import predict_plaintext
from server.state_store import save_twin_state, get_twin_state, BACKEND

app = FastAPI(title="TwinVEIL Inference Service")


class PredictRequest(BaseModel):
    vehicle_id: str
    # Exactly 4 normalised floats in [-1, 1]; enforced at the boundary so
    # bad payloads return 422 Unprocessable Entity instead of a 500.
    features: Annotated[list[float], Field(min_length=4, max_length=4)]


@app.post("/predict")
def predict(req: PredictRequest):
    result = predict_plaintext(req.features)
    save_twin_state(req.vehicle_id, result)
    return result


@app.get("/state/{vehicle_id}")
def state(vehicle_id: str):
    return get_twin_state(vehicle_id) or {"status": "no state"}


@app.get("/health")
def health():
    return {"status": "ok", "redis_backend": BACKEND}


# Make the package importable as `server.main` whether launched as
# `uvicorn server.main:app` from twinveil-server/ or imported in tests.
