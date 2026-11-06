#!/usr/bin/env python
"""Run reconstruction inference for one numpy sample or a numpy dataset."""

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
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default="outputs/predictions/inference_prediction.npy")
    parser.add_argument("--plot-path", default="outputs/reconstructions/inference_prediction.png")
    parser.add_argument("--target-path", default=None, help="Optional target array for plotting comparison")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", choices=["trace", "apd"], default="trace")
    args = parser.parse_args()

    import torch
    from src.evaluation.evaluate import predict
    from src.models.reconstruction_model import build_model
    from src.visualization.plot_reconstruction import plot_reconstruction

    raw_inputs = np.load(args.input_path)
    inputs = add_channel_axis(raw_inputs)
    target_length = None
    if args.task == "apd":
        target_length = 15
    model = build_model(1, 32, 7, 5, target_length=target_length, input_length=inputs.shape[1])
    model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"), strict=True)
    preds = predict(model, inputs, batch_size=args.batch_size, device=args.device)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, preds)

    if args.task == "trace":
        target = None
        if args.target_path:
            target = np.load(args.target_path)[args.sample_index]
        else:
            target = preds[args.sample_index]
        plot_reconstruction(
            inputs[args.sample_index, :, 0],
            target,
            preds[args.sample_index],
            output_path=args.plot_path,
            title=f"Inference sample {args.sample_index}",
        )
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
