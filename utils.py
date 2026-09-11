"""Small training/inference helpers shared by the entry points."""

import json
import os
import random
from copy import deepcopy

import numpy as np
import torch


class EMA:
    """Exponential moving average of model weights.

    The averaged copy is what gets checkpointed and used for inference; it is
    noticeably more stable than the raw weights for diffusion models.

    Args:
        beta: Decay factor; higher means slower, smoother tracking.
    """

    def __init__(self, beta: float = 0.9999):
        self.beta = beta

    def update_model_average(self, ma_model: torch.nn.Module, current_model: torch.nn.Module):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = self.update_average(ma_params.data, current_params.data)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def clone_for_ema(model: torch.nn.Module) -> torch.nn.Module:
    """Return a detached copy of ``model`` to hold the EMA weights."""
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def linear_decay(
    current_epoch: int,
    start_weight: float = 1.0,
    end_weight: float = 0.0001,
    total_epochs: int = 100,
    start_epoch: int = 0,
) -> float:
    """Linearly anneal a loss weight from ``start_weight`` down to ``end_weight``.

    Used for the real-data supervision term ``lambda(t)``: the imperfectly aligned
    real pairs help bridge the synthetic-to-real domain gap early in training, but
    their misalignment noise should not be propagated later on.

    Args:
        current_epoch: Current epoch index.
        start_weight: Weight before annealing starts.
        end_weight: Weight at (and after) the final epoch.
        total_epochs: Total number of training epochs.
        start_epoch: Epoch at which annealing begins.

    Returns:
        The weight for ``current_epoch``.
    """
    if current_epoch < start_epoch:
        return start_weight

    remaining_epochs = total_epochs - start_epoch
    if remaining_epochs <= 0:
        return end_weight

    decay_rate = (start_weight - end_weight) / remaining_epochs
    current_weight = start_weight - decay_rate * (current_epoch - start_epoch)
    return max(current_weight, end_weight)


def set_seed(seed: int = 10):
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_performance(num_threads: int = 4):
    """Enable the throughput settings used for the reported experiments."""
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(num_threads)
    torch.autograd.set_detect_anomaly(False)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def load_config(path: str) -> dict:
    """Load a JSON config file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(config: dict, device: str = "cuda"):
    """Instantiate the TDBM bridge and its conditional U-Net from a config dict.

    ``in_channel`` is ``neighbor_num + 1``: the noisy bridge state plus the
    ``neighbor_num`` conditioning frames of the temporal window.
    """
    from models import BrownianBridgeModel, UNet

    unet = UNet(
        image_size=config["imgSize"],
        in_channel=config["neighbor_num"] + 1,
        inner_channel=config.get("inner_channel", 64),
        out_channel=1,
        res_blocks=config.get("res_blocks", 2),
        attn_res=config.get("attn_res", [16]),
    )
    model = BrownianBridgeModel(
        unet,
        num_timesteps=config.get("num_timesteps", 1000),
        sample_step=config.get("sample_step", 20),
        image_size=config["imgSize"],
    )
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str = "cuda") -> int:
    """Load ``model_state_dict`` from a checkpoint and return the stored epoch."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", -1)
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch + 1})")
    return epoch
