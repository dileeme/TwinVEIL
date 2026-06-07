"""
twinveil-server/server/main.py — FastAPI entry point for TwinVEIL.

Endpoints
---------
GET  /health                 liveness + redis status
POST /predict                plaintext inference (Module 2 contract)
POST /upload_context         store client's public CKKS eval context
POST /predict_he             homomorphic inference (Module 1 / Module 3)
GET  /state/{vehicle_id}     last prediction for a vehicle (Redis, TTL 3600 s)
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .inference import predict_plaintext, predict_encrypted

# ── Redis (real if available, fakeredis fallback) ─────────────────────────────

def _make_redis():
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True, socket_connect_timeout=1)
        r.ping()
        return r, "real"
    except Exception:
        try:
            import fakeredis
            return fakeredis.FakeRedis(decode_responses=True), "fake"
        except ImportError:
            return None, "none"


_redis_client = None
_redis_mode = "fake"

STATE_TTL = 3600  # seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_client, _redis_mode
    _redis_client, _redis_mode = _make_redis()
    yield


app = FastAPI(title="TwinVEIL", version="1.0.0", lifespan=lifespan)

# ── in-process store for the public CKKS evaluation context ──────────────────
# Key: "default" (single shared context for now; Module 3 may extend to per-vehicle)
_pub_ctx_store: dict[str, str] = {}


# ── request / response models ─────────────────────────────────────────────────

class PlaintextRequest(BaseModel):
    vehicle_id: str
    features: list[float]   # 4 normalised floats, order: speed/accel/steering/brake


class PlaintextResponse(BaseModel):
    vehicle_id: str
    anomaly_score: float
    label: int


class UploadContextRequest(BaseModel):
    pub_ctx: str            # base-64 serialised TenSEAL public context


class HERequest(BaseModel):
    vehicle_id: str
    ciphertext: str         # base-64 serialised CKKSVector


class HEResponse(BaseModel):
    vehicle_id: str
    encrypted_result: str   # base-64 serialised CKKSVector


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_state(vehicle_id: str, payload: dict) -> None:
    if _redis_client is None:
        return
    try:
        import json
        _redis_client.setex(f"twin:{vehicle_id}", STATE_TTL, json.dumps(payload))
    except Exception:
        pass


def _load_state(vehicle_id: str) -> Optional[dict]:
    if _redis_client is None:
        return None
    try:
        import json
        raw = _redis_client.get(f"twin:{vehicle_id}")
        return json.loads(raw) if raw else None
    except Exception:
        return None


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "redis": _redis_mode}


@app.post("/predict", response_model=PlaintextResponse)
def predict(req: PlaintextRequest):
    if len(req.features) != 4:
        raise HTTPException(status_code=422, detail="features must have exactly 4 elements")
    result = predict_plaintext(req.features)
    payload = {"vehicle_id": req.vehicle_id, **result}
    _save_state(req.vehicle_id, payload)
    return payload


@app.post("/upload_context")
def upload_context(req: UploadContextRequest):
    _pub_ctx_store["default"] = req.pub_ctx
    return {"status": "ok"}


@app.post("/predict_he", response_model=HEResponse)
def predict_he(req: HERequest):
    pub_ctx = _pub_ctx_store.get("default")
    if pub_ctx is None:
        raise HTTPException(
            status_code=400,
            detail="No public context uploaded. Call POST /upload_context first.",
        )
    encrypted_result = predict_encrypted(req.ciphertext, pub_ctx)
    return {"vehicle_id": req.vehicle_id, "encrypted_result": encrypted_result}


@app.get("/state/{vehicle_id}")
def get_state(vehicle_id: str):
    state = _load_state(vehicle_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No state for vehicle {vehicle_id!r}")
    return state
