def get_loss(name="smooth_l1"):
    import torch

    name = name.lower()
    if name in {"smooth_l1", "huber"}:
        return torch.nn.SmoothL1Loss()
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "mae":
        return torch.nn.L1Loss()
    raise ValueError(f"Unknown loss: {name}")
