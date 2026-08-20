"""Benchmark orchestrator: trains models via Hydra multirun, then evaluates each.

Usage examples:

    # Scaling law: train 6 configs x 3 seeds, then evaluate each
    python -m scdino.benchmark.run \
      --train-args "+experiment=scaling datamodule.loader.max_train_samples=100,500,1000,5000,10000,50000 seed=42,43,44" \
      --inference-args "eval.k=20"

    # Re-evaluate previously trained models with different kNN settings
    python -m scdino.benchmark.run \
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
    """Find the *run* directories under *root* that hold an exported model.

    Returns the run directory (the one containing `hf_model/`), not the
    `hf_model/` directory itself: callers append "hf_model" to build the
    checkpoint path and write the evaluation alongside it.
    """
    return sorted(p.parent.parent for p in root.rglob("hf_model/config.json"))


def run_command(cmd: list[str], label: str) -> bool:
    """Run a command, print output, return True on success."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
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
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Sweep directory. Passed to training as hydra.sweep.dir and used as "
            "the root for model discovery, so the orchestrator knows exactly "
            "where its own outputs are. Defaults to the sweep dir baked into "
            "the experiment config, which must then be given explicitly when "
            "using --skip-training"
        ),
    )
    parser.add_argument(
        "--train-output-dir",
        type=str,
        default=None,
        help="Deprecated alias for --output-dir",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training stage; requires --output-dir",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.train_output_dir
    if args.skip_training and not output_dir:
        parser.error("--skip-training requires --output-dir")

    # --- Stage 1: Training ---------------------------------------------------
    if not args.skip_training:
        train_cmd = [
            sys.executable,
            "-m",
            "scdino.train.run",
            "--multirun",
            *args.train_args.split(),
        ]
        if output_dir:
            train_cmd.append(f"hydra.sweep.dir={output_dir}")
        if not run_command(train_cmd, "Stage 1: Training"):
            # Continuing would evaluate whatever models happen to be on disk
            # from an earlier sweep and report them as this run's results.
            print("[ERROR] Training failed; refusing to evaluate stale models.")
            sys.exit(1)

    # --- Discover trained models ----------------------------------------------
    if not output_dir:
        print(
            "[ERROR] --output-dir was not given, so the models this run produced "
            "cannot be told apart from earlier sweeps under outputs/benchmark/. "
            "Re-run with --output-dir pointing at the sweep directory."
        )
        sys.exit(1)

    output_root = Path(output_dir)
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
        # Write the evaluation next to the model it evaluated. Without this it
        # lands in outputs/inference/<date>/<time>/, disconnected from the sweep,
        # and benchmark.collect never sees it. No timestamp, so re-running the
        # stage overwrites rather than accumulating orphans.
        inference_cmd = [
            sys.executable,
            "-m",
            "scdino.inference.run",
            f"local_model_path={hf_path}",
            f"hydra.run.dir={(model_dir / 'inference').as_posix()}",
            *args.inference_args.split(),
        ]
        ok = run_command(inference_cmd, label)
        (succeeded if ok else failed).append(model_dir.name)

    # --- Summary --------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"  Models found:      {len(model_dirs)}")
    print(f"  Inference success:  {len(succeeded)}")
    print(f"  Inference failed:   {len(failed)}")
    if failed:
        print(f"  Failed runs: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    main()
