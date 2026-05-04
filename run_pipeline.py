"""
Syntho-Limb NS2C Engine: Full Pipeline Runner
==============================================
Runs all pipeline stages in dependency order.

Stages:
  scrub_video       Strip Apple/camera metadata from raw footage
  generate_kin      Extract 7-DOF kinematics via MediaPipe (or fallback)
  tribe_extraction  Generate synthetic fMRI tensor with TRIBE v2
  neural_masking    Apply BA4/BA6 motor cortex mask (atlas + fallback)
  verify            Sanity-check tensor shapes and alignment
  train             Train LSTM decoder + LOO evaluation vs. linear baseline
  simulate          Run MuJoCo simulation with smoothed trajectory

Usage:
  python run_pipeline.py                        # full run
  python run_pipeline.py --from_stage train     # resume from a specific stage
  python run_pipeline.py --only verify          # run one stage
  python run_pipeline.py --config custom.yaml   # use a different config
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGES: list[tuple[str, str]] = [
    ("scrub_video",      "scripts/scrub_video.py"),
    ("generate_kin",     "scripts/generate_kinematics.py"),
    ("tribe_extraction", "scripts/tribe_extraction.py"),
    ("neural_masking",   "scripts/neural_masking.py"),
    ("verify",           "scripts/verify_tensors.py"),
    ("train",            "scripts/train_decoder.py"),
    ("simulate",         "scripts/simulate_robot.py"),
]

STAGE_NAMES = [s[0] for s in STAGES]


def run_stage(name: str, script: str, config: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  STAGE: {name}")
    print(f"{'=' * 60}")
    if not Path(script).exists():
        print(f"[!] Script not found: {script} — skipping.")
        return True  # non-fatal

    # Scripts that don't accept --config yet just get called without it
    cmd = [sys.executable, script, "--config", config]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[!] Stage '{name}' exited with code {result.returncode}.")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Syntho-Limb pipeline end-to-end.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--from_stage",
        default=None,
        metavar="STAGE",
        help=f"Skip all stages before STAGE.  Choices: {STAGE_NAMES}",
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="STAGE",
        help="Run only this single stage.",
    )
    args = parser.parse_args()

    if args.from_stage and args.from_stage not in STAGE_NAMES:
        print(f"[!] Unknown stage '{args.from_stage}'. Available: {STAGE_NAMES}")
        sys.exit(1)
    if args.only and args.only not in STAGE_NAMES:
        print(f"[!] Unknown stage '{args.only}'. Available: {STAGE_NAMES}")
        sys.exit(1)

    start_idx = STAGE_NAMES.index(args.from_stage) if args.from_stage else 0

    for name, script in STAGES[start_idx:]:
        if args.only and name != args.only:
            continue
        if not run_stage(name, script, args.config):
            print("\n[!] Pipeline aborted.")
            sys.exit(1)

    print("\n[*] Pipeline complete.")
