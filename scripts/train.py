#!/usr/bin/env python
"""Train the 1D U-Net reconstruction model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import add_channel_axis, load_default_task
from src.utils.io import load_config, resolve_path
from src.utils.seed import set_seed


def get_nested(config, path, default=None):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", choices=["extra_to_intra", "intra_to_extra", "apd"], default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    task = args.task or get_nested(config, "data.task", "extra_to_intra")
    seed = args.seed if args.seed is not None else get_nested(config, "training.seed", 42)
    set_seed(seed)

    if args.input_path and args.target_path:
        inputs = add_channel_axis(np.load(resolve_path(args.input_path)))
        targets = add_channel_axis(np.load(resolve_path(args.target_path)))
    else:
        inputs, targets = load_default_task(task)

    from src.training.train import prepare_split, train_model

    test_size = get_nested(config, "data.val_size", 0.3 if task != "apd" else 0.6)
    train_input, train_target, val_input, val_target, _, _ = prepare_split(inputs, targets, test_size=test_size, seed=seed)
    print(f"train_input:{train_input.shape}, val_input:{val_input.shape}, train_target:{train_target.shape}, val_target:{val_target.shape}")

    target_length = targets.shape[1] if task == "apd" else None
    checkpoint_path = args.checkpoint_path or get_nested(config, "training.checkpoint_path", "checkpoints/ION_train_checkpoint.pt")
    metrics_path = get_nested(config, "training.log_path", "outputs/metrics/training_log.csv")
    loss_name = args.loss or get_nested(config, "training.loss", "mse" if task == "apd" else "smooth_l1")

    train_model(
        train_input,
        train_target,
        val_input,
        val_target,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size or get_nested(config, "training.batch_size", 32),
        epochs=args.epochs or get_nested(config, "training.epochs", 30),
        lr=args.lr or get_nested(config, "training.lr", 0.0001),
        weight_decay=get_nested(config, "training.weight_decay", 1e-5),
        loss_name=loss_name,
        device=args.device or get_nested(config, "training.device", None),
        target_length=target_length,
        input_length=inputs.shape[1],
        metrics_path=metrics_path,
    )


if __name__ == "__main__":
    main()
