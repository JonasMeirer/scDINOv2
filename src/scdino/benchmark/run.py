"""Benchmark orchestrator: trains models via Hydra multirun, then evaluates each.

Usage examples:

    # Scaling law: train 6 configs x 3 seeds, then evaluate each
    python -m src.scdino.benchmark.run \
      --train-args "+experiment=scaling datamodule.loader.max_train_samples=100,500,1000,5000,10000,50000 seed=42,43,44" \
      --inference-args "eval.k=20"

    # Re-evaluate previously trained models with different kNN settings
    python -m src.scdino.benchmark.run \
      --skip-training \
      --train-output-dir outputs/benchmark/scaling/ \
      --inference-args "eval.k=50 eval.T=0.1"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_hf_models(root: Path) -> list[Path]:
    """Recursively find all hf_model/ directories under *root*."""
    return sorted(p.parent for p in root.rglob("hf_model/config.json"))


def run_command(cmd: list[str], label: str) -> bool:
    """Run a command, print output, return True on success."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] {label} exited with code {result.returncode}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark orchestrator")
    parser.add_argument(
        "--train-args",
        type=str,
        default="",
        help="Hydra overrides for training (space-separated, as one string)",
    )
    parser.add_argument(
        "--inference-args",
        type=str,
        default="",
        help="Extra Hydra overrides for inference (space-separated, as one string)",
    )
    parser.add_argument(
        "--train-output-dir",
        type=str,
        default=None,
        help="Root dir containing trained hf_model/ dirs (skips training)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training stage; requires --train-output-dir",
    )
    args = parser.parse_args()

    if args.skip_training and not args.train_output_dir:
        parser.error("--skip-training requires --train-output-dir")

    # --- Stage 1: Training ---------------------------------------------------
    if not args.skip_training:
        train_cmd = [
            sys.executable,
            "-m",
            "src.scdino.train.run",
            "--multirun",
            *args.train_args.split(),
        ]
        run_command(train_cmd, "Stage 1: Training")

    # --- Discover trained models ----------------------------------------------
    if args.train_output_dir:
        output_root = Path(args.train_output_dir)
    else:
        # Infer sweep dir from the train args by looking for the most recent
        # outputs/benchmark/ directory.
        output_root = Path("outputs/benchmark")

    model_dirs = find_hf_models(output_root)
    if not model_dirs:
        print(f"[ERROR] No hf_model/ directories found under {output_root}")
        sys.exit(1)

    print(f"\nFound {len(model_dirs)} trained model(s):")
    for d in model_dirs:
        print(f"  {d / 'hf_model'}")

    # --- Stage 2: Inference ---------------------------------------------------
    succeeded = []
    failed = []
    for model_dir in model_dirs:
        hf_path = str((model_dir / "hf_model").resolve())
        label = f"Inference: {model_dir.name}"
        inference_cmd = [
            sys.executable,
            "-m",
            "src.scdino.inference.run",
            f"local_model_path={hf_path}",
            *args.inference_args.split(),
        ]
        ok = run_command(inference_cmd, label)
        (succeeded if ok else failed).append(model_dir.name)

    # --- Summary --------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Benchmark Summary")
    print(f"{'='*60}")
    print(f"  Models found:      {len(model_dirs)}")
    print(f"  Inference success:  {len(succeeded)}")
    print(f"  Inference failed:   {len(failed)}")
    if failed:
        print(f"  Failed runs: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    main()
