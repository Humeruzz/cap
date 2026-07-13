import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed for Python random, numpy, and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on availability."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
