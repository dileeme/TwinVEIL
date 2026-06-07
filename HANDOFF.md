# TwinVEIL — Module 1 Handoff

**Purpose:** Resume building Module 1 (CKKS Encrypted Telemetry Pipeline) with full knowledge of what Module 2 produced and what Module 3 expects.

---

## Session Progress — 2026-06-07

### Files Created ✅

| File | Status | Notes |
|------|--------|-------|
| `configs/ckks.yaml` | ✅ Done | CKKS params, model paths, server URL |
| `client/encryptor.py` | ✅ Done | normalise → encrypt → decrypt; secret key stays in process |
| `client/telemetry_client.py` | ✅ Done | HTTP client; upload context, POST ciphertext, decrypt result; CLI included |
| `twinveil-server/server/inference.py` | ✅ Done | `predict_plaintext()` (Module 2 unchanged) + `predict_encrypted()` (HE path) |

### Key implementation notes
- CKKS context created once via `get_context()`; secret key never serialised to disk or sent over wire.
- `public_context_b64()` in `encryptor.py` produces the eval-only context for server upload.
- Poly sigmoid used server-side: `σ(t) ≈ 0.5 + 0.197t − 0.004t³` (degree-3, valid ≈ [-5,5]).
- `predict_encrypted()` in `inference.py` takes `(ciphertext_b64, pub_ctx_b64)` and returns `encrypted_result_b64`.

---

## Next Steps (pick up here)

### 1. Install TenSEAL
```powershell
$env:PIP_CONFIG_FILE = "D:\TWIN\TwinVEIL\.pip\pip.ini"
& "D:\TWIN\TwinVEIL\.venv\Scripts\pip.exe" install tenseal
# If no Python 3.12 wheel: try --pre, or create a 3.11 venv
```

### 2. Add HE endpoints to the FastAPI server
In `twinveil-server/server/main.py` (or wherever the router lives), add:

```python
# POST /upload_context  — store pub_ctx in app state
# POST /predict_he      — call inference.predict_encrypted(ciphertext, pub_ctx)
#                         return {"vehicle_id": ..., "encrypted_result": ...}
```
Request/response shapes:
- `/upload_context`: `{"pub_ctx": "<b64>"}` → `{"status": "ok"}`
- `/predict_he`: `{"vehicle_id": str, "ciphertext": "<b64>"}` → `{"vehicle_id": str, "encrypted_result": "<b64>"}`

### 3. Smoke test end-to-end
```powershell
# Terminal 1 — start server
$py = "D:\TWIN\TwinVEIL\.venv\Scripts\python.exe"
cd D:\TWIN\TwinVEIL\twinveil-server
& $py -m uvicorn server.main:app --reload

# Terminal 2 — send an encrypted reading
cd D:\TWIN\TwinVEIL\TwinVEIL_module1\client
& $py telemetry_client.py --upload-ctx --speed 20 --accel -4 --steering 100 --brake 0.8
# Expected: label=1 (hard-brake-at-speed anomaly rule)

& $py telemetry_client.py --speed 10 --accel 0.5 --steering 30 --brake 0.1
# Expected: label=0 (normal)
```

### 4. Verify HE score matches plaintext baseline
HE anomaly score should be within ~1-2 % of `predict_plaintext()` output for the same normalised inputs. Run a quick comparison script across a sample of `data/processed/labelled.csv` rows.

### 5. Hand off to Module 3
Once the smoke test passes, Module 3 (6G netem evaluation) can begin. It needs:
- Working `/predict_he` endpoint
- `models/lr_weights.npy`, `lr_bias.npy` (already exist — do not retrain)
- `data/processed/labelled.csv` (already exists)
- Plaintext baseline: accuracy 0.9974, F1 0.9759, AUC 0.9991

---

## Project Location

| Item | Path |
|------|------|
| Project root | `D:\TWIN\TwinVEIL` |
| Virtual env | `D:\TWIN\TwinVEIL\.venv` (Python 3.12.5) |
| All packages | D: drive only — nothing on C: |

```powershell
$env:PIP_CONFIG_FILE = "D:\TWIN\TwinVEIL\.pip\pip.ini"
$py = "D:\TWIN\TwinVEIL\.venv\Scripts\python.exe"
```

---

## System Overview

```
Vehicle / Edge Device (Module 1 — YOUR TARGET)
        │
        │  CKKS Encryption (TenSEAL)
        ▼
Encrypted Telemetry (ciphertext bytes over HTTP)
        │
        ▼
Cloud / MEC — FastAPI server (Module 2 — COMPLETE)
        │  encrypted dot-product + poly sigmoid
        ▼
Encrypted Result (ciphertext)
        │
        ▼
Trusted Domain — decrypt & display (Module 1 — YOUR TARGET)
        │
        ▼
Module 3 — 6G netem emulation + evaluation (NOT YET BUILT)
```

---

## Module 2 — What Was Built (Complete ✅)

### What it does
Plaintext anomaly-detection baseline for connected-vehicle telemetry. Trains a logistic-regression model and serves it via FastAPI + Redis.

### Inputs consumed by Module 2

| Input | Format | Source |
|-------|--------|--------|
| POST `/predict` body | `{"vehicle_id": str, "features": [f1,f2,f3,f4]}` | client `encryptor.py` |
| Feature vector | 4 × float64, normalised to [-1, 1] | `models/scaler_bounds.json` |

Feature order (fixed, canonical): `speed, longitudinal_accel, steering_angle, brake_pressure`

### Outputs produced by Module 2

| Output | Format | Notes |
|--------|--------|-------|
| POST `/predict` response | `{"anomaly_score": float, "label": 0\|1}` | |
| GET `/state/{vehicle_id}` | same dict as last prediction | Redis key `twin:{vehicle_id}`, TTL 3600 s |
| GET `/health` | `{"status": "ok", "redis": "real"\|"fake"}` | |
| `models/lr_weights.npy` | shape (4,) float64 | **HE contract — do not retrain** |
| `models/lr_bias.npy` | scalar float64 | **HE contract — do not retrain** |
| `models/scaler_bounds.json` | `{feature: [lo, hi]}` | fixed domain bounds for normalisation |

### Trained model weights (the HE contract)

```
w = [-16.1164, 1.5204, 28.4368, 21.7180]   # speed, accel, steering, brake
b =  11.8909
```

Plaintext inference: `score = sigmoid(w · x + b)`

Verified: manual sigmoid reproduces sklearn `predict_proba` to < 1e-5.

### Normalisation bounds (`models/scaler_bounds.json`)

| Feature | lo | hi |
|---------|----|----|
| speed | 0 | 40 |
| longitudinal_accel | -8 | 5 |
| steering_angle | -500 | 500 |
| brake_pressure | 0 | 1 |

Formula: `x_norm = clip(2*(x - lo)/(hi - lo) - 1, -1, 1)`

### Key design decision: the single-file swap contract

- **Server side:** only `twinveil-server/server/inference.py` changes for HE. `predict_plaintext()` becomes `predict_encrypted()`. No other server file is touched.
- **Client side:** only `client/encryptor.py` changes. `normalise()` becomes `encrypt()`. Payload becomes serialised ciphertext bytes.

### Anomaly rules (for reference / dataset regeneration)

| Rule | Condition | Label |
|------|-----------|-------|
| Hard-brake-at-speed | `speed_raw > 15 AND accel_raw < -3.5` | 1 |
| Steering oscillation | `\|steering_raw\| > 200 AND \|accel_raw\| < 0.5` | 1 |
| Accel spike | `accel_raw > 3.0 AND speed_raw < 5` | 1 |

Anomaly rate: 5.25 %. Dataset: 299,620 rows (synthetic, schema-identical to comma2k19 target schema).

---

## Module 1 — What You Need to Build

### Goal
Stand up the CKKS encryption pipeline on the trusted client side, swap the server's plaintext inference for homomorphic inference, so that the cloud never sees plaintext telemetry.

### CKKS parameters (from spec §)

```python
import tenseal as ts

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=2**14,        # 16 384
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()
```

### Files to create / modify

| File | Action | What it does |
|------|--------|-------------|
| `client/encryptor.py` | **Replace** `normalise()` → `encrypt()` | Normalise raw readings, then encrypt into a CKKS ciphertext; serialise for HTTP transport |
| `twinveil-server/server/inference.py` | **Add** `predict_encrypted()` | Load weights, compute `w·x_enc + b` homomorphically + degree-3 poly sigmoid; return ciphertext |
| `configs/ckks.yaml` | **Create** | Store CKKS params + paths for context/keys |
| `client/telemetry_client.py` | **Create** | Send serialised ciphertext to `/predict`, receive encrypted result, decrypt locally |

### Degree-3 polynomial sigmoid approximation

The sigmoid cannot be evaluated directly on ciphertexts. Use:

```
σ(x) ≈ 0.5 + 0.197x − 0.004x³
```

This is the standard Chebyshev-based degree-3 fit valid over roughly [-5, 5]. Because `w·x + b` on normalised inputs has a known bounded range, this approximation is safe.

Implemented homomorphically as:
```python
# x_enc is the encrypted dot product (w·x + b)
result = 0.5 + 0.197 * x_enc - 0.004 * (x_enc ** 3)
```

### Server endpoint change for HE

`POST /predict` currently expects `{"vehicle_id": str, "features": [float×4]}`.

For Module 1/3, change to accept `{"vehicle_id": str, "ciphertext": "<base64 bytes>"}`, perform `predict_encrypted()`, return `{"vehicle_id": str, "encrypted_result": "<base64 bytes>"}`. The client decrypts with the private key (which never leaves the trusted domain).

### Key constraint: private key stays client-side

The CKKS context with secret key is created once in the trusted domain. The server receives only a **public evaluation context** (no secret key). Serialise contexts:

```python
# Client keeps
secret_ctx = context  # full context with secret key

# Send to server (no secret key)
pub_ctx_bytes = context.serialize(save_secret_key=False)
```

---

## Module 3 — What Will Be Built After Module 1

### Goal
Simulate 6G network conditions (Linux `tc netem`) and evaluate end-to-end latency, accuracy vs plaintext, and ciphertext overhead.

### Inputs it needs from Modules 1 & 2

| Input | Comes from |
|-------|-----------|
| Working encrypted `/predict` endpoint | Module 1 |
| `models/lr_weights.npy`, `lr_bias.npy` | Module 2 |
| `data/processed/labelled.csv` (test rows) | Module 2 |
| Plaintext metrics baseline (accuracy 0.9974, F1 0.9759, AUC 0.9991) | Module 2 |

### Metrics to capture

| Metric | Target |
|--------|--------|
| End-to-end latency (plaintext vs encrypted) | measure & compare |
| Accuracy under HE | within ~1–2 % of plaintext |
| Ciphertext overhead (bytes vs plaintext payload) | measure |
| Throughput under simulated 6G RTT / BW | measure |

### `tc netem` setup (Linux only — needs WSL or Linux VM)

```bash
# Example: 5G/6G-ish latency profile
sudo tc qdisc add dev lo root netem delay 10ms 2ms distribution normal loss 0.01%
```

---

## Installed Packages (D: venv — relevant to Module 1)

| Package | Version | Notes |
|---------|---------|-------|
| tenseal | install needed | CKKS HE — **not yet installed** |
| numpy | 1.26.4 | ✅ |
| fastapi | 0.111.0 | ✅ |
| uvicorn | 0.30.1 | ✅ |
| redis | 5.0.4 | ✅ |
| fakeredis | latest | ✅ fallback |
| requests | installed | ✅ (client) |

Install TenSEAL:
```powershell
$env:PIP_CONFIG_FILE = "D:\TWIN\TwinVEIL\.pip\pip.ini"
& "D:\TWIN\TwinVEIL\.venv\Scripts\pip.exe" install tenseal
```

> Note: TenSEAL wheels exist for Python 3.10/3.11. If 3.12 has no prebuilt wheel, use `--pre` or install from source, or pin to Python 3.11 in a new venv.

---

## Artefacts from Module 2 (do not delete or retrain)

```
models/lr_weights.npy          ← HE contract (weights)
models/lr_bias.npy             ← HE contract (bias)
models/scaler_bounds.json      ← normalisation bounds
models/lr_baseline.pkl         ← sklearn model (plaintext fallback)
data/processed/labelled.csv    ← labelled dataset for Module 3 eval
reports/step3_metrics.txt      ← plaintext baseline metrics
```

---

## Quick Sanity Check Before Starting

Run the existing Module 2 server and confirm it responds:

```powershell
$py = "D:\TWIN\TwinVEIL\.venv\Scripts\python.exe"
cd D:\TWIN\TwinVEIL\twinveil-server
Start-Process $py -ArgumentList "step4_test.py"
# Or run the full test suite:
& $py step4_test.py
```

All 4 assertions should still pass (label=0, label=1, state round-trip, doc anomaly vector).
