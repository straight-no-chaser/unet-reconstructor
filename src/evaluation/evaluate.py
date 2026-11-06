from pathlib import Path

import numpy as np

from src.evaluation.metrics import mae_per_sample, mse_per_sample, summarize
from src.models.reconstruction_model import build_model
from src.utils.io import ensure_dir, save_json
from src.visualization.plot_reconstruction import plot_reconstruction_grid


def predict(model, inputs, batch_size=32, device=None):
    import torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    x = torch.tensor(np.transpose(inputs, (0, 2, 1)), dtype=torch.float)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (x_batch,) in loader:
            y_pred = model(x_batch.to(device))
            preds.append(np.asarray(y_pred.cpu())[:, 0, :])
    return np.concatenate(preds)


def evaluate_checkpoint(
    checkpoint_path,
    inputs,
    targets,
    output_dir="outputs",
    batch_size=32,
    device=None,
    target_length=None,
    input_length=None,
    strict=True,
):
    import torch

    input_length = input_length or inputs.shape[1]
    model = build_model(1, 32, 7, 5, target_length=target_length, input_length=input_length)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=strict)
    preds = predict(model, inputs, batch_size=batch_size, device=device)
    target_2d = targets[:, :, 0] if targets.ndim == 3 else targets

    output_dir = Path(output_dir)
    ensure_dir(output_dir / "metrics")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "reconstructions")

    np.save(output_dir / "predictions" / "predictions.npy", preds)
    mse = mse_per_sample(preds, target_2d)
    mae = mae_per_sample(preds, target_2d)
    metrics = {"mse": summarize(mse), "mae": summarize(mae), "n_samples": int(len(preds))}
    save_json(metrics, output_dir / "metrics" / "evaluation_metrics.json")
    plot_reconstruction_grid(inputs[:, :, 0], target_2d, preds, output_dir / "reconstructions" / "sample_reconstructions.png")
    return metrics, preds
