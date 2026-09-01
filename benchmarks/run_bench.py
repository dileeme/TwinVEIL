"""
benchmarks/run_bench.py — run the whole Module 3 evaluation suite.

Executes the three benchmarks in one process so the expensive CKKS keygen
(~5 s) and the model load happen once rather than three times, then prints a
consolidated summary of the numbers the paper needs.

Usage:
    python benchmarks/run_bench.py                 # defaults
    python benchmarks/run_bench.py -n 30           # more samples everywhere
    python benchmarks/run_bench.py --latency-n 8   # override just the slow one
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from common import RESULTS, banner, bootstrap, environment

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accuracy_delta          # noqa: E402
import latency_bench           # noqa: E402
import overhead                # noqa: E402


def _run(module, argv: list[str], label: str) -> bool:
    saved = sys.argv
    sys.argv = [label] + argv
    try:
        return module.main() == 0
    except Exception as exc:                     # pragma: no cover
        print(f"  ! {label} failed: {type(exc).__name__}: {exc}")
        return False
    finally:
        sys.argv = saved


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Module 3 evaluation suite")
    ap.add_argument("-n", type=int, default=20, help="default sample count")
    ap.add_argument("--latency-n", type=int, default=None,
                    help="samples for the latency benchmark (slowest; defaults to n/2)")
    args = ap.parse_args()
    lat_n = args.latency_n if args.latency_n is not None else max(4, args.n // 2)

    banner("TwinVEIL Module 3 — evaluation suite")
    t0 = time.perf_counter()
    bs = bootstrap()
    print(f"  CKKS keygen (shared across all three benchmarks): {bs['keygen_s']:.2f} s")

    ok = {
        "overhead":       _run(overhead, ["-n", str(args.n)], "overhead"),
        "accuracy_delta": _run(accuracy_delta, ["-n", str(args.n)], "accuracy_delta"),
        "latency":        _run(latency_bench, ["-n", str(lat_n)], "latency_bench"),
    }

    # ── consolidated headline numbers ────────────────────────────────────
    banner("Headline numbers")
    try:
        ov = json.loads((RESULTS / "overhead.json").read_text())["meta"]
        ac = json.loads((RESULTS / "accuracy_delta.json").read_text())["meta"]
        la = json.loads((RESULTS / "latency.json").read_text())["meta"]

        warm = next(s for s in la["stage_summary_ms"] if "WARM" in s["stage"])
        cold = next(s for s in la["stage_summary_ms"] if "COLD" in s["stage"])

        print(f"  {'accuracy cost of encryption (CKKS noise)':<48}"
              f"{ac['ckks_noise']['mean']:>14.2e}")
        print(f"  {'score gap from POLY_SCALE rescaling':<48}"
              f"{ac['approx_delta']['mean']:>14.4f}")
        print(f"  {'label agreement, encrypted vs plaintext':<48}"
              f"{ac['label_agreement'] * 100:>13.2f}%")
        print(f"  {'ciphertext expansion (on the wire)':<48}"
              f"{ov['expansion_vs_features']:>13,.0f}x")
        print(f"  {'sustained uplink per vehicle @10 Hz':<48}"
              f"{ov['uplink_mbps_encrypted']:>11,.1f} Mbit/s")
        print(f"  {'evaluation context, as built':<48}"
              f"{ov['context_bytes_with_galois'] / 2**20:>11,.0f} MiB")
        print(f"  {'evaluation context, without unused Galois keys':<48}"
              f"{ov['context_bytes_without_galois'] / 2**20:>11,.0f} MiB")
        print(f"  {'HE inference, as implemented (cold context)':<48}"
              f"{cold['mean_ms']:>11,.0f} ms")
        print(f"  {'HE inference, context cached (achievable)':<48}"
              f"{warm['mean_ms']:>11,.0f} ms")
    except Exception as exc:
        print(f"  (could not consolidate: {exc})")

    banner("Summary")
    for name, passed in ok.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n  total wall-clock: {time.perf_counter() - t0:.1f} s")
    print(f"  results in {RESULTS}")
    print(f"  environment: {environment()['platform']}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
