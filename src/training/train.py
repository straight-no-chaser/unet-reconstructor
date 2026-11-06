import csv
import time
from pathlib import Path

import numpy as np

from src.data.dataset import make_dataloaders, train_val_split_indices
from src.models.reconstruction_model import build_model
from src.training.losses import get_loss
from src.utils.io import ensure_dir


def prepare_split(inputs, targets, test_size=0.3, seed=None):
    idx_train, idx_val = train_val_split_indices(len(inputs), test_size=test_size, seed=seed)
    return inputs[idx_train], targets[idx_train], inputs[idx_val], targets[idx_val], idx_train, idx_val


def train_model(
    train_input,
    train_target,
    val_input,
    val_target,
    checkpoint_path,
    batch_size=32,
    epochs=30,
    lr=0.0001,
    weight_decay=1e-5,
    loss_name="smooth_l1",
    device=None,
    target_length=None,
    input_length=None,
    metrics_path=None,
):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    input_length = input_length or train_input.shape[1]
    model = build_model(1, 32, 7, 5, target_length=target_length, input_length=input_length).to(device)
    loss_fn = get_loss(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.8, min_lr=1e-8)
    train_loader, valid_loader, _ = make_dataloaders(train_input, train_target, val_input, val_target, batch_size=batch_size)

    checkpoint_path = Path(checkpoint_path)
    ensure_dir(checkpoint_path.parent)
    best_val_loss = float("inf")
    history = []

    train_preds = np.zeros(int(train_input.shape[0] * train_target.shape[1]))
    val_preds = np.zeros(int(val_input.shape[0] * val_target.shape[1]))
    train_targets = np.zeros(int(train_input.shape[0] * train_target.shape[1]))

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.0

        for i, (x_batch, y_batch) in enumerate(train_loader):
            y_pred = model(x_batch.to(device))
            loss = loss_fn(y_pred.cpu(), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

            pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
            start = i * batch_size * train_target.shape[1]
            end = (i + 1) * batch_size * train_target.shape[1]
            train_preds[start:end] = pred.reshape((-1))
            train_targets[start:end] = y_batch.detach().cpu().argmax(axis=1).reshape((-1))

        model.eval()
        avg_val_loss = 0.0
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch.to(device))
                avg_val_loss += loss_fn(y_pred.cpu(), y_batch).item() / len(valid_loader)
                pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
                start = i * batch_size * val_target.shape[1]
                end = (i + 1) * batch_size * val_target.shape[1]
                val_preds[start:end] = pred.reshape((-1))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

        try:
            train_score = f1_score(train_targets, train_preds, average="macro")
            val_score = f1_score(val_target.argmax(axis=2).reshape((-1)), val_preds, average="macro")
        except Exception:
            train_score = float("nan")
            val_score = float("nan")

        elapsed_time = time.time() - start_time
        scheduler.step(avg_val_loss)
        row = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "val_loss": avg_val_loss,
            "train_f1_legacy": train_score,
            "val_f1_legacy": val_score,
            "seconds": elapsed_time,
        }
        history.append(row)
        print(
            "Epoch {}/{} \t loss={:.4f} \t train_f1={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} \t time={:.2f}s".format(
                epoch + 1, epochs, avg_loss, train_score, avg_val_loss, val_score, elapsed_time
            )
        )

    if metrics_path:
        metrics_path = Path(metrics_path)
        ensure_dir(metrics_path.parent)
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    return model, history
