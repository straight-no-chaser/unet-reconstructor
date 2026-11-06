import numpy as np


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:
        pass


def load_npy(path, mmap_mode=None):
    return np.load(path, mmap_mode=mmap_mode)


def add_channel_axis(array):
    array = np.asarray(array)
    if array.ndim == 2:
        return array.reshape(array.shape[0], array.shape[1], 1)
    return array


def train_val_split_indices(n_samples, test_size=0.3, seed=None):
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = int(round(n_samples * test_size))
    return indices[n_val:], indices[:n_val]


def load_trace_arrays(input_path, target_path):
    inputs = add_channel_axis(np.load(input_path))
    targets = add_channel_axis(np.load(target_path))
    if len(inputs) != len(targets):
        raise ValueError(f"Input and target sample counts differ: {len(inputs)} vs {len(targets)}")
    return inputs, targets


def load_default_task(task, root="."):
    from pathlib import Path

    root = Path(root)
    processed = root / "data" / "processed"
    legacy_inputs = root / "inputs"
    inputs_dir = processed / "inputs" if (processed / "inputs").exists() else legacy_inputs

    if task == "extra_to_intra":
        input_path = processed / "all_extras.npy" if (processed / "all_extras.npy").exists() else root / "all_extras.npy"
        target_path = processed / "all_intras.npy" if (processed / "all_intras.npy").exists() else root / "all_intras.npy"
        return load_trace_arrays(input_path, target_path)
    if task == "intra_to_extra":
        input_path = processed / "all_intras.npy" if (processed / "all_intras.npy").exists() else root / "all_intras.npy"
        target_path = processed / "all_extras.npy" if (processed / "all_extras.npy").exists() else root / "all_extras.npy"
        inputs, targets = load_trace_arrays(input_path, target_path)
        return inputs, targets
    if task == "apd":
        extras = add_channel_axis(np.load(inputs_dir / "training inputs" / "extras_new_normalization.npy"))
        apds = add_channel_axis(np.load(inputs_dir / "training inputs" / "APDS_new_normalization.npy"))
        return extras, apds
    raise ValueError(f"Unknown task: {task}")


class IONDataset(Dataset):
    """Original notebook dataset wrapper.

    Inputs and targets are expected as numpy arrays shaped [N, signal_length, 1].
    The returned tensors preserve the notebook interface: [channels, signal_length].
    """

    def __init__(self, train_input, train_output, mode="train"):
        self.train_input = train_input
        self.train_output = train_output
        self.mode = mode

    def __len__(self):
        return len(self.train_input)

    def _augmentations(self, input_data, target_data):
        if np.random.rand() < 0.5:
            input_data = input_data[::-1]
            target_data = target_data[::-1]
        return input_data, target_data

    def __getitem__(self, idx):
        if torch is None:
            raise ImportError("PyTorch is required to materialize IONDataset samples.")
        x = self.train_input[idx]
        y = self.train_output[idx]
        if self.mode == "train":
            x, y = self._augmentations(x, y)
        out_x = torch.tensor(np.transpose(x.copy(), (1, 0)), dtype=torch.float)
        out_y = torch.tensor(np.transpose(y.copy(), (1, 0)), dtype=torch.float)
        return out_x, out_y


ION_Dataset = IONDataset


def make_dataloaders(train_input, train_target, val_input, val_target, batch_size=32):
    if torch is None:
        raise ImportError("PyTorch is required to create DataLoader objects.")
    train = IONDataset(train_input, train_target, mode="train")
    valid = IONDataset(val_input, val_target, mode="valid")
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    x_test = torch.tensor(np.transpose(val_input, (0, 2, 1)), dtype=torch.float)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test), batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader
