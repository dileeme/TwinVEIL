"""
benchmarks/overhead.py — Module 3, ciphertext and transmission overhead.

Measures what encryption costs on the wire: ciphertext size versus plaintext
payload, total bytes per prediction, and the resulting uplink demand at the
10 Hz telemetry rate Module 2's dataset simulates.

Also sizes the CKKS evaluation context, which is a one-off but a large one and
easy to leave out of an overhead table by accident. It is reported both with and
without Galois keys: client/encryptor.py calls generate_galois_keys(), but the
inference circuit (dot product, square, cube) performs no rotations, so those
keys are never used. The difference is the cheapest available saving and the
paper should report the achievable figure alongside the current one.

Usage:
    python benchmarks/overhead.py
"""
from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import tenseal as ts

from common import (
    PROFILES, banner, bootstrap, environment, load_samples, write_results,
)

TELEMETRY_HZ = 10.0          # Module 2's generator samples at 10 Hz
PLAINTEXT_FEATURE_BYTES = 4 * 8   # 4 x float64


def main() -> int:
    ap = argparse.ArgumentParser(description="Module 3 ciphertext overhead benchmark")
    ap.add_argument("-n", type=int, default=10, help="samples to size (default 10)")
    args = ap.parse_args()

    banner("Module 3 — ciphertext and transmission overhead")
    bs = bootstrap()
    ctx, pub = bs["ctx"], bs["pub_ctx_b64"]
    X, _, source = load_samples(args.n)
    print(f"  samples: {source}\n")

    rows = []
    for i, x in enumerate(X):
        ct_raw = ts.ckks_vector(ctx, x.tolist()).serialize()
        ct_b64 = base64.b64encode(ct_raw).decode()
        res_b64 = bs["predict_encrypted"](ct_b64, pub)
        plain_req = json.dumps({"vehicle_id": "v001", "features": [round(v, 6) for v in x]})
        rows.append({
            "i": i,
            "plaintext_features_bytes": PLAINTEXT_FEATURE_BYTES,
            "plaintext_request_bytes": len(plain_req),
            "ciphertext_raw_bytes": len(ct_raw),
            "ciphertext_b64_bytes": len(ct_b64),
            "result_b64_bytes": len(res_b64),
            "expansion_vs_features": round(len(ct_b64) / PLAINTEXT_FEATURE_BYTES, 1),
            "expansion_vs_request": round(len(ct_b64) / len(plain_req), 1),
            "total_wire_bytes": len(ct_b64) + len(res_b64),
        })

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    ct_b64_mean = col("ciphertext_b64_bytes").mean()
    res_mean = col("result_b64_bytes").mean()
    wire_mean = col("total_wire_bytes").mean()
    plain_wire = col("plaintext_request_bytes").mean() + 64

    print(f"  {'payload':<34}{'bytes':>14}{'vs plaintext':>15}")
    print(f"  {'-' * 63}")
    print(f"  {'telemetry (4 x float64)':<34}{PLAINTEXT_FEATURE_BYTES:>14,}{'1x':>15}")
    print(f"  {'plaintext JSON request':<34}{plain_wire - 64:>14,.0f}"
          f"{(plain_wire - 64) / PLAINTEXT_FEATURE_BYTES:>14.1f}x")
    print(f"  {'ciphertext (raw)':<34}{col('ciphertext_raw_bytes').mean():>14,.0f}"
          f"{col('ciphertext_raw_bytes').mean() / PLAINTEXT_FEATURE_BYTES:>14,.0f}x")
    print(f"  {'ciphertext (base64, on wire)':<34}{ct_b64_mean:>14,.0f}"
          f"{ct_b64_mean / PLAINTEXT_FEATURE_BYTES:>14,.0f}x")
    print(f"  {'encrypted result (base64)':<34}{res_mean:>14,.0f}"
          f"{res_mean / PLAINTEXT_FEATURE_BYTES:>14,.0f}x")
    print(f"  {'total per prediction':<34}{wire_mean:>14,.0f}"
          f"{wire_mean / plain_wire:>14,.0f}x")

    # ── one-off context cost ─────────────────────────────────────────────
    pub_raw = base64.b64decode(pub)
    ctx_no_galois = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree=2 ** 14,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
    )
    ctx_no_galois.global_scale = 2 ** 40
    no_galois = len(ctx_no_galois.serialize(save_secret_key=False))

    print(f"\n  {'one-off evaluation context':<34}{'bytes':>14}")
    print(f"  {'-' * 48}")
    print(f"  {'as built (with Galois keys)':<34}{len(pub_raw):>14,}"
          f"   ({len(pub_raw) / 2**20:,.0f} MiB)")
    print(f"  {'without Galois keys':<34}{no_galois:>14,}"
          f"   ({no_galois / 2**20:,.0f} MiB)")
    print(f"  {'saving (keys are never used)':<34}{len(pub_raw) - no_galois:>14,}"
          f"   ({100 * (1 - no_galois / len(pub_raw)):.1f}% smaller)")

    # ── sustained uplink demand ──────────────────────────────────────────
    print(f"\n  sustained uplink at {TELEMETRY_HZ:.0f} Hz per vehicle")
    print(f"  {'-' * 48}")
    he_bps = ct_b64_mean * 8 * TELEMETRY_HZ
    pt_bps = (plain_wire - 64) * 8 * TELEMETRY_HZ
    print(f"  {'plaintext':<34}{pt_bps / 1e3:>10,.1f} kbit/s")
    print(f"  {'encrypted':<34}{he_bps / 1e6:>10,.1f} Mbit/s")
    print(f"  {'ratio':<34}{he_bps / pt_bps:>10,.0f}x")

    print(f"\n  time to ship ONE ciphertext, by profile")
    print(f"  {'-' * 48}")
    for p in PROFILES:
        if p.uplink_mbps >= 1e5:
            continue
        print(f"  {p.name:<14}{p.uplink_mbps:>6.0f} Mbit/s uplink"
              f"{p.transport_s(int(ct_b64_mean), int(res_mean)) * 1000:>12,.0f} ms")

    meta = {
        "benchmark": "overhead", "n_samples": args.n, "sample_source": source,
        "telemetry_hz": TELEMETRY_HZ,
        "mean_ciphertext_b64_bytes": float(ct_b64_mean),
        "mean_total_wire_bytes": float(wire_mean),
        "expansion_vs_features": float(ct_b64_mean / PLAINTEXT_FEATURE_BYTES),
        "context_bytes_with_galois": len(pub_raw),
        "context_bytes_without_galois": no_galois,
        "context_saving_pct": round(100 * (1 - no_galois / len(pub_raw)), 2),
        "uplink_mbps_encrypted": he_bps / 1e6,
        "uplink_kbps_plaintext": pt_bps / 1e3,
        "note": (
            "Galois keys are generated in client/encryptor.py but the inference "
            "circuit performs no rotations, so the context can be shipped without "
            "them at no functional cost."
        ),
        "environment": environment(),
    }
    c, j = write_results("overhead", rows, meta)
    print(f"\n  wrote {c.name}, {j.name} to {c.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
