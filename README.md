# TwinVEIL

**TwinVEIL** is a privacy-preserving framework for 6G Digital Twins that performs
anomaly detection on **encrypted** vehicle telemetry using CKKS homomorphic
encryption.

Telemetry is encrypted at the edge, inference runs directly on ciphertext in an
untrusted cloud/MEC environment, and plaintext is recovered only inside the
trusted user domain. The secret key never leaves the client.

---

## Overview

Digital Twins rely on continuous telemetry streams to stay synchronized with the
physical systems they mirror. Transport-layer security protects that data in
transit, but telemetry is conventionally **decrypted before processing** in cloud
infrastructure — so the operator sees everything.

TwinVEIL closes that gap: the anomaly-detection model evaluates a logistic
regression and a polynomial activation entirely under encryption, so the cloud
learns neither the telemetry nor the result.

---

## Architecture

```text
Vehicle / Edge Device   ── trusted domain ──┐
        │                                   │
        │ CKKS encryption (secret key here) │
        ▼                                   │
Encrypted telemetry                         │
        │                                   │
        ▼                                   │
Cloud / MEC  ── untrusted ──                │
(encrypted w·x + b, poly sigmoid)           │
        │                                   │
        ▼                                   │
Encrypted result ───────────────────────────┘
        │
        ▼
Decryption & interpretation (trusted domain)
```

---

## Repository layout

This repository is split across two working branches plus an integration tree.

| Branch | Contents |
|--------|----------|
| `Twinveil-module1` | Module 1 — the CKKS encrypted pipeline |
| `Twinveil-module2` | Module 2 — the plaintext ML baseline that trains the model |
| `main` | integration branch |

```text
Twinveil-module1
├── client/
│   ├── encryptor.py            CKKS context, normalise, encrypt, decrypt_result
│   └── telemetry_client.py     HTTP client + CLI
├── twinveil-server/server/
│   ├── main.py                 FastAPI app
│   ├── inference.py            predict_plaintext + predict_encrypted
│   └── poly_approx.py          degree-3 sigmoid approximation
├── models/                     lr_weights.npy, lr_bias.npy, scaler_bounds.json
├── configs/ckks.yaml           reference CKKS parameters (see Known limitations)
├── tests/test_pipeline.py      17 end-to-end tests
└── requirements.txt

Twinveil-module2
├── scripts/                    step1 generate → step1 clean → step2 label → step3 train
├── client/encryptor.py         plaintext normaliser + client
├── twinveil-server/server/     main.py, inference.py, state_store.py
├── models/                     trained artefacts (source of truth for the weights)
├── data/                       generated datasets (gitignored)
├── reports/                    audit, metrics, confusion matrix, test output
├── docs/IMPLEMENTATION.md      Module 2 implementation report
└── run.py                      runs all four steps end to end
```

---

## Modules

### Module 1 — Encrypted telemetry pipeline · **implemented**

* CKKS context generation inside the trusted client domain
* Edge-side normalisation and encryption of the 4-feature telemetry vector
* Public **evaluation** context (no secret key) uploaded to the server
* Encrypted `w·x + b` followed by a degree-3 polynomial activation
* Server returns an encrypted score; only the client can decrypt it

### Module 2 — Plaintext ML baseline · **implemented**

* Synthetic connected-vehicle telemetry generation, audit and cleaning
* Rule-based anomaly labelling and fixed-bounds normalisation to `[-1, 1]`
* Logistic-regression training; exports the weights consumed by Module 1
* FastAPI `/predict` service with Redis-backed twin state

### Module 3 — 6G integration & evaluation · **not started**

Planned: `tc netem` transport emulation, end-to-end latency, throughput and
ciphertext-overhead measurement, and an HE-vs-plaintext accuracy comparison.
No code for this module exists yet.

---

## Quick start

Python 3.12. A prepared virtual environment lives at `TwinModule2/.venv`.

**Module 1 — encrypted pipeline**

```powershell
pip install -r requirements.txt
python -m pytest tests/test_pipeline.py -v
```

Against a live server:

```powershell
uvicorn server.main:app --app-dir twinveil-server --port 8000
python client/telemetry_client.py --upload-ctx `
    --speed 30 --accel -5.0 --steering 8 --brake 0.9
```

`--upload-ctx` sends the public evaluation context; it is required once per
server process before `/predict_he` will accept a ciphertext.

**Module 2 — regenerate the model** (on the `Twinveil-module2` branch)

```powershell
python run.py
```

This runs data generation → cleaning → labelling → training → endpoint tests, and
writes `models/lr_weights.npy` and `models/lr_bias.npy`.

### API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | liveness + active Redis backend |
| `POST` | `/upload_context` | store the client's public CKKS evaluation context |
| `POST` | `/predict_he` | **encrypted** inference; returns an encrypted score |
| `POST` | `/predict` | plaintext inference (Module 2 baseline) |
| `GET` | `/state/{vehicle_id}` | last stored twin state |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Scheme | CKKS (Microsoft SEAL via TenSEAL) |
| `poly_modulus_degree` | 16384 (2¹⁴) |
| `coeff_mod_bit_sizes` | `[60, 40, 40, 40, 40, 60]` |
| `global_scale` | 2⁴⁰ |
| Multiplicative depth used | 3 of 4 available levels |
| Features | `speed`, `longitudinal_accel`, `steering_angle`, `brake_pressure` |

Activation: `P(z) = 0.5 + 0.198285·z − 0.004469·z³`, least-squares fitted to the
sigmoid on `[-5, 5]`. The logit is divided by `POLY_SCALE = 16` before evaluation
to bring it inside that interval.

---

## Results

**Module 1** — 17/17 tests pass (~23 s), covering the encryption round trip,
encrypted inference through the FastAPI app, and label agreement between the
encrypted and plaintext paths on every test sample.

**Module 2** — logistic regression on a held-out 20% split (59,924 rows):

| Metric | Value |
|--------|------:|
| Accuracy | 0.9974 |
| Precision (anomaly) | 0.9558 |
| Recall (anomaly) | 0.9968 |
| F1 (anomaly) | 0.9759 |
| ROC-AUC | 0.9991 |

Please read these alongside the first limitation below — they characterise the
dataset as much as the classifier.

---

## Known limitations

These are open issues, recorded here so results are not read as stronger than
they are.

**1. The synthetic dataset is circular.** `comma2k19` proved unusable (torrent-only
distribution; `brake_pressure` and `longitudinal_accel` are both absent), so the
data is generated. The generator seeds driving events at the exact index sets that
the labelling rules then fire on, and anomalous steering was made one-directional
so a linear model could separate it. The metrics above therefore measure a
construction, not a detection method. The train/test split is also random rather
than temporal, despite the data being a 10 Hz time series.

**2. The encrypted score is not the plaintext probability.** Because the logit is
divided by `POLY_SCALE = 16` before the polynomial, the decrypted value is
`P(z/16) ≈ sigmoid(z/16)`, not `sigmoid(z)`. The **label is preserved** (the sign
is unchanged), but absolute scores diverge from the plaintext path by up to ~0.43.
Treat the encrypted output as a binary decision, not a calibrated probability, and
do not compare the two scores numerically without correcting for the scaling.

**3. `POLY_SCALE` is not tied to the trained weights.** `P(z) > 0.5` only for
`0 < z < 6.661`, and the current worst-case scaled logit is
`(Σ|w| + |b|)/16 = 4.98` — a margin that holds for *these* weights only. Retraining
to larger weights would silently invert labels on extreme inputs. A guard asserting
`(Σ|w| + |b|) / POLY_SCALE < 6.661` at import time is needed.

**4. No authentication or transport security.** `/upload_context` is unauthenticated
and stores into a single shared slot, so any caller can replace the evaluation
context for all clients. There is no TLS, replay protection, or rate limiting, and
`vehicle_id` travels in plaintext. CKKS provides confidentiality but no integrity —
a malicious server could return an arbitrary ciphertext and the client has no way to
verify the computation. **This is a research prototype; do not deploy it as is.**

**5. Twin state is plaintext and encrypted-path-only in name.** `/predict` stores
the plaintext score and label in Redis. `/predict_he` currently stores nothing.

**6. `configs/ckks.yaml` is not read by any code.** The parameters are hardcoded in
`client/encryptor.py`; the YAML file is reference documentation until it is wired up.

**7. Model artefacts have drifted between branches.** The `.npy` weights on this
branch are not bit-identical to the Module 2 export (differences ~1e-5). Regenerate
from `Twinveil-module2` for reproducible results.

**8. `data/` is gitignored.** A fresh clone of `Twinveil-module2` must run `run.py`
before `step4_test.py` will pass, since the test reads `data/processed/labelled.csv`.

---

## Technical stack

| Component | Technology | Status |
|-----------|------------|--------|
| Homomorphic encryption | TenSEAL 0.3.16 (Microsoft SEAL) | in use |
| Cryptographic scheme | CKKS | in use |
| Machine learning | scikit-learn 1.5.0 | in use |
| Backend | FastAPI 0.111 + Uvicorn | in use |
| State storage | Redis 5.0.4 (fakeredis fallback) | in use |
| Network simulation | Linux `tc netem` | planned (Module 3) |
| Language | Python 3.12 | — |

---

## Status

* [x] Scope definition
* [x] System architecture
* [x] CKKS pipeline implementation (Module 1)
* [x] Encrypted inference integration (Module 1)
* [x] Plaintext ML baseline (Module 2)
* [ ] Polynomial-approximation hardening (limitations 2 and 3)
* [ ] Non-circular dataset / temporal evaluation (limitation 1)
* [ ] Network simulation (Module 3)
* [ ] Experimental evaluation (Module 3)
* [ ] IEEE GLOBECOM workshop submission

---

## Citation

If you use TwinVEIL in academic work, please cite the forthcoming paper:

> TwinVEIL: A Privacy-Preserving Framework for 6G Digital Twins, with Applications
> in Vehicular, Industrial, and Smart Infrastructure Systems.

---

## License

MIT License
