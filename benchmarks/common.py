"""
benchmarks/common.py — shared harness for the Module 3 evaluation suite.

Module 3 produces the paper's numbers: end-to-end latency, accuracy delta versus
plaintext, and ciphertext overhead, across a range of 6G transport profiles.

WHAT IS MEASURED VS MODELLED
----------------------------
Compute stages (encrypt, context load, homomorphic evaluate, decrypt) are
measured directly on this machine with perf_counter.

Transport is MODELLED, not measured. The procedure calls for Linux `tc netem`,
which does not exist on Windows, and no netem-capable host is available here.
Rather than fabricate a transport measurement, each profile applies an explicit
analytic model to the real, measured payload sizes:

    transport_s = rtt_s + bytes_up / uplink_bps + bytes_down / downlink_bps

Every emitted row carries a `transport_model` column saying so, and totals are
split into `compute_s` and `transport_s` so a reader can discard the modelled
half. When Dilen's netem rig is available, replace model_transport() with real
timings; nothing else in the suite needs to change.

This mirrors how Module 2 handled the comma2k19 substitution: take the honest
deviation, label it clearly, keep everything else faithful.
"""
from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

# ── locate the sibling module checkouts ──────────────────────────────────────

HERE = Path(__file__).resolve().parent
TWIN_MAIN = HERE.parent
WORKSPACE = TWIN_MAIN.parent
MODULE1 = WORKSPACE / "TwinModule1" / "TwinVEIL"
MODULE2 = WORKSPACE / "TwinModule2"
RESULTS = TWIN_MAIN / "results"

for p in (MODULE1 / "client", MODULE1 / "twinveil-server"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ── transport profiles ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Profile:
    """A 6G transport condition.

    rtt_ms       round-trip time added to every request
    uplink_mbps  edge -> MEC capacity (ciphertext direction; the expensive one)
    downlink_mbps MEC -> edge capacity
    """
    name: str
    rtt_ms: float
    uplink_mbps: float
    downlink_mbps: float
    note: str

    def transport_s(self, bytes_up: int, bytes_down: int) -> float:
        return (
            self.rtt_ms / 1000.0
            + bytes_up * 8 / (self.uplink_mbps * 1e6)
            + bytes_down * 8 / (self.downlink_mbps * 1e6)
        )


# SCOPE NOTE: the network simulation itself (tc netem) is Dilen's half of Module 3.
# These entries are the profile *parameters* the harness runs against — the
# 10 / 50 / 100 ms RTT points the procedure names, plus an `ideal` compute-only
# control so the transport contribution can be read off by subtraction. Bandwidth
# figures are placeholders pending Dilen's rig; when it exists, these values come
# from it and model_transport() is replaced by the measured link.
PROFILES: list[Profile] = [
    Profile("ideal",    0.0,   1e6,  1e6, "compute-only control, no transport cost"),
    Profile("mec-edge", 10.0,  100.0, 200.0, "on-premise MEC, one hop"),
    Profile("metro",    50.0,   50.0, 100.0, "metro aggregation site"),
    Profile("regional", 100.0,  25.0,  50.0, "regional datacentre"),
]

PROFILES_BY_NAME = {p.name: p for p in PROFILES}


# ── CKKS bootstrap (expensive — do it once per process) ──────────────────────

_BOOT: dict | None = None


def bootstrap() -> dict:
    """Build the CKKS context and load the model. Cached for the process.

    Returns a dict of the client/server callables plus the serialized public
    context, so each benchmark shares one keygen (~5 s) instead of paying it
    repeatedly.
    """
    global _BOOT
    if _BOOT is not None:
        return _BOOT

    t0 = time.perf_counter()
    from encryptor import (  # noqa: E402
        get_context, encrypt, decrypt_result, public_context_b64, normalise,
    )
    from server.inference import (  # noqa: E402
        predict_encrypted, predict_plaintext, _load_weights,
    )
    from server.poly_approx import POLY_A0, POLY_A1, POLY_A3, POLY_SCALE  # noqa: E402

    ctx = get_context()
    pub_ctx_b64 = public_context_b64()
    keygen_s = time.perf_counter() - t0

    w, b = _load_weights()
    _BOOT = dict(
        ctx=ctx, pub_ctx_b64=pub_ctx_b64, keygen_s=keygen_s,
        encrypt=encrypt, decrypt_result=decrypt_result, normalise=normalise,
        predict_encrypted=predict_encrypted, predict_plaintext=predict_plaintext,
        w=w, b=float(b),
        poly=dict(A0=POLY_A0, A1=POLY_A1, A3=POLY_A3, SCALE=POLY_SCALE),
    )
    return _BOOT


def poly_sigmoid_scaled(z: float) -> float:
    """Exactly what the encrypted path computes: P(z / POLY_SCALE)."""
    p = bootstrap()["poly"]
    t = z / p["SCALE"]
    return p["A0"] + p["A1"] * t + p["A3"] * t ** 3


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


# ── samples ──────────────────────────────────────────────────────────────────

FEATURES = ["speed_norm", "longitudinal_accel_norm",
            "steering_angle_norm", "brake_pressure_norm"]


def load_samples(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (X, y, source) — n normalised feature vectors and their labels.

    Prefers real rows from Module 2's labelled.csv, stratified so anomalies are
    represented. That file is gitignored, so on a fresh clone we fall back to a
    uniform sample over the normalised domain and say so in `source`.
    """
    csv_path = MODULE2 / "data" / "processed" / "labelled.csv"
    if csv_path.is_file():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, usecols=FEATURES + ["label"])
            half = max(1, n // 2)
            anom = df[df.label == 1].sample(min(half, int((df.label == 1).sum())),
                                            random_state=seed)
            norm = df[df.label == 0].sample(n - len(anom), random_state=seed)
            out = (pd.concat([norm, anom])
                     .sample(frac=1.0, random_state=seed)
                     .reset_index(drop=True))
            return (out[FEATURES].to_numpy(np.float64),
                    out["label"].to_numpy(int),
                    f"labelled.csv ({len(out)} rows, {int(out.label.sum())} anomalies)")
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"  ! could not read {csv_path.name} ({exc}); using synthetic samples")

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 4))
    bs = bootstrap()
    y = np.array([int(sigmoid(float(x @ bs["w"] + bs["b"])) >= 0.5) for x in X])
    return X, y, "synthetic uniform [-1,1]^4 (labelled.csv absent)"


# ── result output (CSV + JSON — the doc left this open; emit both) ───────────

def write_results(name: str, rows: list[dict], meta: dict) -> tuple[Path, Path]:
    RESULTS.mkdir(exist_ok=True)
    csv_path, json_path = RESULTS / f"{name}.csv", RESULTS / f"{name}.json"

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2, default=str)
    return csv_path, json_path


def environment() -> dict:
    """Machine facts every result file should carry for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "tc_netem_available": shutil.which("tc") is not None,
        "transport": "MODELLED (see benchmarks/common.py docstring)",
    }


def timed(fn: Callable) -> tuple[object, float]:
    """Run fn(), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def summarise(label: str, values: list[float], unit: str = "ms") -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "stage": label,
        f"mean_{unit}": round(float(a.mean()), 3),
        f"median_{unit}": round(float(np.median(a)), 3),
        f"p95_{unit}": round(float(np.percentile(a, 95)), 3),
        f"min_{unit}": round(float(a.min()), 3),
        f"max_{unit}": round(float(a.max()), 3),
    }
