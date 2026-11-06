#!/usr/bin/env python
"""Evaluate a reconstruction checkpoint on numpy input/target arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import add_channel_axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default="checkpoints/ION_train_checkpoint.pt")
    parser.add_argument("--input-path", default="data/processed/all_extras.npy")
    parser.add_argument("--target-path", default="data/processed/all_intras.npy")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", choices=["trace", "apd"], default="trace")
    parser.add_argument("--non-strict", action="store_true", help="Allow missing/unexpected checkpoint keys")
    args = parser.parse_args()

    inputs = add_channel_axis(np.load(args.input_path))
    targets = add_channel_axis(np.load(args.target_path))
    target_length = targets.shape[1] if args.task == "apd" else None

    from src.evaluation.evaluate import evaluate_checkpoint

    metrics, _ = evaluate_checkpoint(
        args.checkpoint_path,
        inputs,
        targets,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        target_length=target_length,
        input_length=inputs.shape[1],
        strict=not args.non_strict,
    )
    print(metrics)


if __name__ == "__main__":
    main()
