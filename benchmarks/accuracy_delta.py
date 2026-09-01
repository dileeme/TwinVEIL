"""
benchmarks/accuracy_delta.py — Module 3, encrypted vs plaintext accuracy.

The procedure calls this "a key paper figure": run the same N samples through
both paths, record score deltas and label agreement.

READ THIS BEFORE QUOTING THE SCORE DELTA
----------------------------------------
The two paths do not compute the same function, and the difference is not CKKS
noise. server/inference.py divides the logit by POLY_SCALE = 16 before applying
the degree-3 polynomial, so:

    plaintext path  ->  sigmoid(z)
    encrypted path  ->  P(z / 16)  ~  sigmoid(z / 16)

The label (the sign) survives; the score does not. This benchmark therefore
decomposes the total score gap into two independent components:

    approximation delta = |P(z/16) - sigmoid(z)|   deterministic, by construction
    ckks noise          = |HE decrypted - P(z/16)| the actual cost of encryption

Reporting the total as "accuracy loss caused by homomorphic encryption" would be
wrong by three to four orders of magnitude — the noise term is ~1e-6 while the
approximation term reaches ~0.4. Both numbers are emitted separately so the paper
can state each honestly.

Label agreement is the metric that is actually meaningful end-to-end, and it is
reported with the margin to the decision boundary so near-boundary behaviour is
visible rather than averaged away.

Usage:
    python benchmarks/accuracy_delta.py -n 40
"""
from __future__ import annotations

import argparse
import base64

import numpy as np
import tenseal as ts

from common import (
    banner, bootstrap, environment, load_samples, poly_sigmoid_scaled, sigmoid,
    write_results,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Module 3 accuracy-delta benchmark")
    ap.add_argument("-n", type=int, default=40, help="samples (default 40)")
    args = ap.parse_args()

    banner("Module 3 — accuracy delta: encrypted vs plaintext")
    bs = bootstrap()
    X, y, source = load_samples(args.n)
    print(f"  samples: {source}\n")

    ctx, pub, w, b = bs["ctx"], bs["pub_ctx_b64"], bs["w"], bs["b"]
    rows = []
    for i, (x, label) in enumerate(zip(X, y)):
        ct = base64.b64encode(ts.ckks_vector(ctx, x.tolist()).serialize()).decode()
        he = float(bs["decrypt_result"](bs["predict_encrypted"](ct, pub)))

        z = float(x @ w + b)
        plain = sigmoid(z)
        exact_poly = poly_sigmoid_scaled(z)

        rows.append({
            "i": i,
            "true_label": int(label),
            "logit": round(z, 6),
            "plain_score": round(plain, 8),
            "exact_poly_score": round(exact_poly, 8),
            "he_score": round(he, 8),
            "approx_delta": round(abs(exact_poly - plain), 8),
            "ckks_noise": round(abs(he - exact_poly), 12),
            "total_delta": round(abs(he - plain), 8),
            "plain_label": int(plain >= 0.5),
            "he_label": int(he >= 0.5),
            "label_match": int((plain >= 0.5) == (he >= 0.5)),
            "boundary_margin": round(abs(z), 6),
        })
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(X)}", flush=True)

    approx = np.array([r["approx_delta"] for r in rows])
    noise = np.array([r["ckks_noise"] for r in rows])
    total = np.array([r["total_delta"] for r in rows])
    match = np.array([r["label_match"] for r in rows])

    print(f"\n  {'component':<34}{'mean':>13}{'max':>13}")
    print(f"  {'-' * 60}")
    print(f"  {'CKKS noise (cost of encryption)':<34}{noise.mean():>13.2e}{noise.max():>13.2e}")
    print(f"  {'poly approx (POLY_SCALE=16)':<34}{approx.mean():>13.4f}{approx.max():>13.4f}")
    print(f"  {'total |HE - plaintext|':<34}{total.mean():>13.4f}{total.max():>13.4f}")
    print(f"\n  approximation term is {approx.mean() / noise.mean():.0f}x the noise term")

    print(f"\n  label agreement : {match.sum()}/{len(rows)} ({100 * match.mean():.2f}%)")
    dis = [r for r in rows if not r["label_match"]]
    if dis:
        print("  disagreements (closest to the decision boundary first):")
        for r in sorted(dis, key=lambda r: r["boundary_margin"])[:5]:
            print(f"    i={r['i']:<4} logit={r['logit']:+.4f}  "
                  f"plain={r['plain_score']:.4f} -> {r['plain_label']}   "
                  f"he={r['he_score']:.4f} -> {r['he_label']}")
    else:
        print("  no disagreements")

    # Accuracy against ground truth, both paths.
    yt = np.array([r["true_label"] for r in rows])
    pl = np.array([r["plain_label"] for r in rows])
    hl = np.array([r["he_label"] for r in rows])
    print(f"\n  accuracy vs ground truth: plaintext {(pl == yt).mean():.4f}   "
          f"encrypted {(hl == yt).mean():.4f}")

    meta = {
        "benchmark": "accuracy_delta", "n_samples": args.n, "sample_source": source,
        "poly": {k: float(v) for k, v in bs["poly"].items()},
        "ckks_noise": {"mean": float(noise.mean()), "max": float(noise.max())},
        "approx_delta": {"mean": float(approx.mean()), "max": float(approx.max())},
        "total_delta": {"mean": float(total.mean()), "max": float(total.max())},
        "label_agreement": float(match.mean()),
        "accuracy_plaintext": float((pl == yt).mean()),
        "accuracy_encrypted": float((hl == yt).mean()),
        "interpretation": (
            "The encrypted path computes P(z/POLY_SCALE), not sigmoid(z). Quote "
            "ckks_noise as the cost of encryption; approx_delta is a deterministic "
            "consequence of the logit rescaling, not of homomorphic evaluation."
        ),
        "environment": environment(),
    }
    c, j = write_results("accuracy_delta", rows, meta)
    print(f"\n  wrote {c.name}, {j.name} to {c.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
