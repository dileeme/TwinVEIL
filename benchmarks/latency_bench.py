"""
benchmarks/latency_bench.py — Module 3, end-to-end latency.

Runs N samples through both the plaintext and encrypted paths and records the
per-stage breakdown the procedure asks for: encrypt / transmit / infer / decrypt,
plus label-match rate against the plaintext path.

Two things this measures that are worth knowing before reading the totals:

1.  COLD vs WARM inference. server/inference.py:predict_encrypted() calls
    ts.context_from() on every request, re-deserializing the ~180 MB public
    evaluation context each time. That dominates the measured cost by more than
    an order of magnitude. The harness reports both:
      cold — what the server does today (deserialize + evaluate)
      warm — evaluate only, i.e. what it would cost with the context cached
    The gap is the single largest available optimisation and belongs in the paper.

2.  Transport is MODELLED from the measured payload sizes (see common.py). The
    `compute_ms` and `transport_ms` columns are kept separate so the modelled
    half can be discarded or replaced with netem measurements.

Usage:
    python benchmarks/latency_bench.py            # N=20, all profiles
    python benchmarks/latency_bench.py -n 50
    python benchmarks/latency_bench.py --profiles mec-edge,regional
"""
from __future__ import annotations

import argparse
import base64
import time

import numpy as np
import tenseal as ts

from common import (
    PROFILES, PROFILES_BY_NAME, banner, bootstrap, environment, load_samples,
    sigmoid, summarise, timed, write_results,
)


def measure_samples(X: np.ndarray, y: np.ndarray) -> list[dict]:
    """Per-sample compute timings and correctness, independent of any profile."""
    bs = bootstrap()
    ctx, pub = bs["ctx"], bs["pub_ctx_b64"]
    w, b = bs["w"], bs["b"]

    # Warm path: deserialize the evaluation context once, as a fixed server
    # should. Used to separate crypto cost from the per-request context reload.
    pub_bytes = base64.b64decode(pub)
    warm_ctx = ts.context_from(pub_bytes)

    rows = []
    for i, (x, label) in enumerate(zip(X, y)):
        # ── plaintext path ───────────────────────────────────────────────
        pt, t_plain = timed(lambda: bs["predict_plaintext"](x.tolist()))

        # ── encrypted path ───────────────────────────────────────────────
        ct, t_enc = timed(
            lambda: base64.b64encode(ts.ckks_vector(ctx, x.tolist()).serialize()).decode()
        )

        # cold: exactly what the server does today
        out_cold, t_cold = timed(lambda: bs["predict_encrypted"](ct, pub))

        # warm: same arithmetic, context already resident
        def _warm():
            v = ts.lazy_ckks_vector_from(base64.b64decode(ct))
            v.link_context(warm_ctx)
            p = bs["poly"]
            lin = v.dot((w / p["SCALE"]).tolist()) + b / p["SCALE"]
            sq = lin * lin
            cub = sq * lin
            return (lin * p["A1"] + cub * p["A3"] + p["A0"]).serialize()

        _, t_warm = timed(_warm)

        he_score, t_dec = timed(lambda: bs["decrypt_result"](out_cold))

        z = float(x @ w + b)
        rows.append({
            "i": i,
            "true_label": int(label),
            "logit": round(z, 6),
            "plain_score": round(sigmoid(z), 6),
            "he_score": round(float(he_score), 6),
            "plain_label": int(sigmoid(z) >= 0.5),
            "he_label": int(he_score >= 0.5),
            "label_match": int((sigmoid(z) >= 0.5) == (he_score >= 0.5)),
            "t_plain_infer_ms": round(t_plain * 1000, 4),
            "t_encrypt_ms": round(t_enc * 1000, 3),
            "t_infer_cold_ms": round(t_cold * 1000, 3),
            "t_infer_warm_ms": round(t_warm * 1000, 3),
            "t_decrypt_ms": round(t_dec * 1000, 3),
            "ct_bytes": len(ct),
            "result_bytes": len(out_cold),
            "plain_req_bytes": len(str(x.tolist())),
        })
        if (i + 1) % 5 == 0:
            print(f"    {i + 1}/{len(X)} samples", flush=True)
    return rows


def compose(rows: list[dict], profiles) -> list[dict]:
    """Combine measured compute with each profile's modelled transport."""
    out = []
    for p in profiles:
        for r in rows:
            tr_he = p.transport_s(r["ct_bytes"], r["result_bytes"])
            tr_pt = p.transport_s(r["plain_req_bytes"], 64)
            for mode, compute_ms, transport_s in (
                ("plaintext",   r["t_plain_infer_ms"], tr_pt),
                ("encrypted-cold",
                 r["t_encrypt_ms"] + r["t_infer_cold_ms"] + r["t_decrypt_ms"], tr_he),
                ("encrypted-warm",
                 r["t_encrypt_ms"] + r["t_infer_warm_ms"] + r["t_decrypt_ms"], tr_he),
            ):
                out.append({
                    "profile": p.name,
                    "rtt_ms": p.rtt_ms,
                    "mode": mode,
                    "i": r["i"],
                    "compute_ms": round(compute_ms, 3),
                    "transport_ms": round(transport_s * 1000, 3),
                    "total_ms": round(compute_ms + transport_s * 1000, 3),
                    "label_match": r["label_match"],
                    "transport_model": "analytic: rtt + bytes/bandwidth",
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Module 3 end-to-end latency benchmark")
    ap.add_argument("-n", type=int, default=20, help="samples (default 20)")
    ap.add_argument("--profiles", default="", help="comma-separated subset")
    args = ap.parse_args()

    profiles = ([PROFILES_BY_NAME[n] for n in args.profiles.split(",")]
                if args.profiles else PROFILES)

    banner("Module 3 — end-to-end latency")
    bs = bootstrap()
    print(f"  CKKS keygen (once)  : {bs['keygen_s']:.2f} s")
    X, y, source = load_samples(args.n)
    print(f"  samples             : {source}\n")

    t0 = time.perf_counter()
    per_sample = measure_samples(X, y)
    print(f"\n  measured {len(per_sample)} samples in {time.perf_counter() - t0:.1f} s")

    stages = {
        "encrypt (client)":        [r["t_encrypt_ms"] for r in per_sample],
        "HE infer COLD (server)":  [r["t_infer_cold_ms"] for r in per_sample],
        "HE infer WARM (server)":  [r["t_infer_warm_ms"] for r in per_sample],
        "decrypt (client)":        [r["t_decrypt_ms"] for r in per_sample],
        "plaintext infer":         [r["t_plain_infer_ms"] for r in per_sample],
    }
    stage_rows = [summarise(k, v) for k, v in stages.items()]

    print(f"\n  {'stage':<24}{'mean':>10}{'median':>10}{'p95':>10}  (ms)")
    for s in stage_rows:
        print(f"  {s['stage']:<24}{s['mean_ms']:>10.3f}{s['median_ms']:>10.3f}{s['p95_ms']:>10.3f}")

    cold = float(np.mean(stages["HE infer COLD (server)"]))
    warm = float(np.mean(stages["HE infer WARM (server)"]))
    print(f"\n  context-reload overhead : {cold - warm:.1f} ms/request "
          f"({cold / warm:.0f}x) — removable by caching the evaluation context")

    matches = sum(r["label_match"] for r in per_sample)
    print(f"  label match vs plaintext: {matches}/{len(per_sample)}")

    composed = compose(per_sample, profiles)
    print(f"\n  {'profile':<12}{'mode':<18}{'compute':>10}{'transport':>11}{'total':>10}  (ms)")
    for p in profiles:
        for mode in ("plaintext", "encrypted-cold", "encrypted-warm"):
            sel = [r for r in composed if r["profile"] == p.name and r["mode"] == mode]
            print(f"  {p.name:<12}{mode:<18}"
                  f"{np.mean([r['compute_ms'] for r in sel]):>10.1f}"
                  f"{np.mean([r['transport_ms'] for r in sel]):>11.1f}"
                  f"{np.mean([r['total_ms'] for r in sel]):>10.1f}")

    meta = {
        "benchmark": "latency", "n_samples": args.n, "sample_source": source,
        "keygen_s": round(bs["keygen_s"], 3),
        "stage_summary_ms": stage_rows,
        "context_reload_overhead_ms": round(cold - warm, 3),
        "label_match_rate": matches / len(per_sample),
        "profiles": [
            {"name": p.name, "rtt_ms": p.rtt_ms, "uplink_mbps": p.uplink_mbps,
             "downlink_mbps": p.downlink_mbps, "note": p.note} for p in profiles
        ],
        "environment": environment(),
    }
    c, j = write_results("latency", composed, meta)
    write_results("latency_per_sample", per_sample, meta)
    print(f"\n  wrote {c.name}, {j.name} (+ latency_per_sample.*) to {c.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
