"""Device resolution: CUDA → MPS → CPU."""

from __future__ import annotations

import torch


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(device: torch.device | None = None) -> str:
    device = device or auto_device()
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return f"CUDA:{idx} — {name} ({mem_gb:.1f} GB)"
    if device.type == "mps":
        return "MPS (Apple Silicon)"
    return "CPU"
