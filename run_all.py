"""
run_all.py — TwinVEIL integration harness.

The project is split across three branches of one repository:

    Twinveil-module2   plaintext ML baseline; trains and exports the model
    Twinveil-module1   CKKS encrypted pipeline; consumes that model
    main               this branch: integration + validation

Module 1 does not train anything. Module 2 does not encrypt anything. The seam
between them is a pair of .npy files, and in September 2026 that seam silently
drifted — Module 1 was carrying a 4-decimal transcription of the weights while
Module 2 published metrics for the full-precision export. Step 0 below exists so
that cannot recur unnoticed.

Layout expected (branches checked out as siblings of this one):

    <parent>/
    ├── Twin-main/          <- main             (this tree)
    ├── TwinModule1/
    │   └── TwinVEIL/       <- Twinveil-module1
    └── TwinModule2/        <- Twinveil-module2

Usage:
    python run_all.py             run every stage
    python run_all.py --contract  only the cross-branch model contract check
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

MODULE1 = PARENT / "TwinModule1" / "TwinVEIL"
MODULE2 = PARENT / "TwinModule2"

# Artefacts that must be byte-identical across the two module branches.
CONTRACT_FILES = ["lr_weights.npy", "lr_bias.npy", "scaler_bounds.json"]


# ── environment ──────────────────────────────────────────────────────────────

def find_python() -> str:
    """Locate an interpreter with tenseal + sklearn installed.

    Prefers the project venv (Module 2 owns the only one); falls back to whatever
    is running this script. Deliberately not a hardcoded absolute path — the
    previous version of this file pinned one and broke on every other machine.
    """
    for candidate in (
        MODULE2 / ".venv" / "Scripts" / "python.exe",   # Windows
        MODULE2 / ".venv" / "bin" / "python",           # POSIX
    ):
        if candidate.is_file():
            return str(candidate)
    print(f"  ! project venv not found under {MODULE2 / '.venv'}; "
          f"falling back to {sys.executable}")
    return sys.executable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── stage 0: cross-branch model contract ─────────────────────────────────────

def check_contract() -> bool:
    """Module 1 and Module 2 must be serving the same model, byte for byte."""
    banner("Step 0  Cross-branch model contract")

    src, dst = MODULE2 / "models", MODULE1 / "models"
    for d in (src, dst):
        if not d.is_dir():
            print(f"  FAIL  missing model directory: {d}")
            return False

    ok = True
    for name in CONTRACT_FILES:
        a, b = src / name, dst / name
        if not a.is_file() or not b.is_file():
            print(f"  FAIL  {name:<20} missing on one side")
            ok = False
            continue
        ha, hb = sha256(a), sha256(b)
        if ha == hb:
            print(f"  ok    {name:<20} {ha[:16]}")
        else:
            print(f"  FAIL  {name:<20} DRIFT")
            print(f"          module2 {ha}")
            print(f"          module1 {hb}")
            ok = False

    if not ok:
        print("\n  Module 1 is serving a different model than Module 2 trained.")
        print(f"  Fix by copying the canonical export verbatim (do not retype):")
        for name in CONTRACT_FILES:
            print(f"      cp {src / name} {dst / name}")
        print(f"  Then refresh the checksums in {MODULE1 / 'models' / 'MANIFEST.json'}.")
    return ok


# ── stages 1-2: the two test suites ──────────────────────────────────────────

def banner(title: str) -> None:
    # flush: these interleave with subprocess output, which bypasses this buffer.
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def run_suite(title: str, cmd: list[str], cwd: Path) -> bool:
    banner(title)
    if not cwd.is_dir():
        print(f"  SKIP  {cwd} not checked out", flush=True)
        return False
    print(f"  $ {' '.join(cmd)}\n    (in {cwd})\n", flush=True)
    return subprocess.run(cmd, cwd=str(cwd)).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="TwinVEIL integration harness")
    ap.add_argument("--contract", action="store_true",
                    help="only run the cross-branch model contract check")
    args = ap.parse_args()

    py = find_python()
    print(f"TwinVEIL integration harness\n  interpreter: {py}\n  workspace  : {PARENT}")

    results: list[tuple[str, bool]] = [("Model contract", check_contract())]

    if not args.contract:
        results.append((
            "Module 1 (encrypted pipeline)",
            run_suite("Step 1  Module 1 - CKKS encrypted pipeline",
                      [py, "-m", "pytest", "tests/", "-q"], MODULE1),
        ))
        results.append((
            "Module 2 (plaintext service)",
            run_suite("Step 2  Module 2 - plaintext inference service",
                      [py, "-m", "pytest", "step4_test.py", "-q"],
                      MODULE2 / "twinveil-server"),
        ))

    banner("Summary")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} stage(s) failed.")
        if "Module 2 (plaintext service)" in failed:
            print("  Note: Module 2's tests read data/processed/labelled.csv, which is")
            print("  gitignored. On a fresh clone run `python run.py` in TwinModule2 first.")
        return 1

    print("\nAll stages passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
