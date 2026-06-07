"""
TwinVEIL — full pipeline runner.

Runs all Module 2 steps in order:
  1. Generate synthetic telemetry
  2. Audit + clean
  3. Feature engineering + labelling
  4. Train logistic-regression baseline
  5. Endpoint tests (Step 4.7)

Usage:
  python run.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("Step 1.1  Generate data",       [PY, "scripts/step1_generate_data.py"]),
    ("Step 1.3/4  Audit + clean",     [PY, "scripts/step1_audit_clean.py"]),
    ("Step 2  Features + labels",     [PY, "scripts/step2_features.py"]),
    ("Step 3  Train LR baseline",     [PY, "scripts/step3_train.py"]),
    ("Step 4.7  Endpoint tests",      [PY, "step4_test.py"]),
]


def run_step(label: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    for label, cmd in STEPS[:-1]:
        run_step(label, cmd, cwd=ROOT)

    # Step 4.7 must run from twinveil-server/ so server imports resolve
    label, cmd = STEPS[-1]
    run_step(label, cmd, cwd=ROOT / "twinveil-server")

    print(f"\n{'='*60}")
    print("  All steps completed successfully.")
    print(f"{'='*60}\n")
