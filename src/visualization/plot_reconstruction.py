from pathlib import Path

import numpy as np


def plot_reconstruction(input_signal, target_signal, prediction, output_path=None, title=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if input_signal is not None:
        ax.plot(input_signal, color="0.5", alpha=0.7, label="Input")
    ax.plot(target_signal, color="b", label="Measured target")
    ax.plot(prediction, color="r", label="Network prediction")
    ax.legend()
    if title:
        ax.set_title(title)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_reconstruction_grid(inputs, targets, predictions, output_path, n_examples=6):
    import matplotlib.pyplot as plt

    n_examples = min(n_examples, len(predictions))
    indices = np.linspace(0, len(predictions) - 1, n_examples, dtype=int)
    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 2.2 * n_examples), sharex=False)
    if n_examples == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        if inputs is not None:
            ax.plot(inputs[idx], color="0.5", alpha=0.7, label="Input")
        ax.plot(targets[idx], color="b", label="Measured target")
        ax.plot(predictions[idx], color="r", label="Network prediction")
        ax.set_title(f"Sample {idx}")
    axes[0].legend(loc="upper right")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
