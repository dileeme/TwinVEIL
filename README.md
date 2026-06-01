# TwinVEIL

**TwinVEIL** is a privacy-preserving framework for 6G Digital Twins that enables secure edge-to-cloud analytics using CKKS homomorphic encryption.

The framework encrypts telemetry data at the edge, performs inference directly on encrypted data in an untrusted cloud or MEC environment, and ensures that plaintext is visible only within the trusted user domain.

---

## Overview

Digital Twins rely on continuous telemetry streams to maintain synchronized virtual representations of physical systems. While conventional transport-layer security protects data in transit, telemetry is typically decrypted before processing in cloud infrastructure.

TwinVEIL addresses this gap by enabling encrypted computation through homomorphic encryption, allowing anomaly detection models to operate without exposing sensitive telemetry data.

---

## Architecture

```text
Vehicle / Edge Device
        │
        │ CKKS Encryption
        ▼
Encrypted Telemetry
        │
        ▼
Cloud / MEC
(Encrypted Inference)
        │
        ▼
Encrypted Results
        │
        ▼
Trusted User Domain
(Decryption & Visualization)
```

### Module 1 — Encrypted Telemetry Pipeline

* CKKS key generation within the trusted domain
* Edge-side telemetry encryption
* Secure ciphertext transmission
* Threat-model enforcement

### Module 2 — Digital Twin Inference Service

* Encrypted logistic regression inference
* Degree-3 polynomial sigmoid approximation
* FastAPI prediction service
* Redis-based encrypted twin state management

### Module 3 — 6G Integration & Evaluation

* Simulated 6G transport using Linux tc netem
* End-to-end latency evaluation
* Accuracy comparison against plaintext inference
* Throughput and ciphertext overhead analysis

---

## Technical Stack

| Component              | Technology     |
| ---------------------- | -------------- |
| Homomorphic Encryption | TenSEAL        |
| Cryptographic Scheme   | CKKS           |
| Machine Learning       | Scikit-learn   |
| Backend                | FastAPI        |
| State Storage          | Redis          |
| Network Simulation     | Linux tc netem |
| Language               | Python         |
| HE Backend             | Microsoft SEAL |

---

## Research Objectives

* Protect Digital Twin telemetry from cloud-side exposure
* Evaluate encrypted inference under realistic network conditions
* Quantify accuracy loss caused by homomorphic computation
* Demonstrate practical privacy-preserving Digital Twin deployment for 6G systems

---

## Repository Structure

```text
TwinVEIL/
│
├── client/
│   ├── encryptor.py
│   └── telemetry_client.py
│
├── server/
│   ├── inference.py
│   ├── poly_approx.py
│   ├── state_store.py
│   └── main.py
│
├── training/
│   └── train_lr.py
│
├── simulator/
│   └── network_sim.py
│
├── configs/
│   └── ckks.yaml
│
├── tests/
│
├── docs/
│
└── README.md
```

---

## Current Status

* [x] Scope definition completed
* [x] System architecture finalized
* [ ] CKKS pipeline implementation
* [ ] Encrypted inference integration
* [ ] Network simulation
* [ ] Experimental evaluation
* [ ] IEEE GLOBECOM workshop submission

---

## Citation

If you use TwinVEIL in academic work, please cite the forthcoming paper:

> TwinVEIL: A Privacy-Preserving Framework for 6G Digital Twins, with Applications in Vehicular, Industrial, and Smart Infrastructure Systems.

---

## License

MIT License
