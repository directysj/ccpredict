"""Choose where PyTorch runs (only the stage-1 CNN uses PyTorch)."""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)
_threads_applied: Optional[int] = None


def select_device(device: str = "cpu", num_threads: Optional[int] = None):
    """Return the torch.device for ``device`` and apply the CPU thread setting.

    device="cpu"   CPU only
    device="gpu"   CUDA if available, else Apple Metal (MPS); raises if neither
    device="auto"  CUDA > MPS > CPU
    """
    import torch

    global _threads_applied
    if device not in ("cpu", "gpu", "auto"):
        raise ValueError(f"device must be 'cpu', 'gpu' or 'auto', not {device!r}")
    if num_threads and num_threads != _threads_applied:
        torch.set_num_threads(num_threads)
        _threads_applied = num_threads

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if device == "cpu":
        dev = torch.device("cpu")
    elif has_cuda:
        dev = torch.device("cuda")
    elif has_mps:
        dev = torch.device("mps")
    elif device == "gpu":
        raise RuntimeError("device='gpu' but PyTorch sees neither CUDA nor Apple MPS; use 'cpu' or 'auto'")
    else:
        dev = torch.device("cpu")
    log.info("PyTorch on %s (device=%r, %d CPU threads)", dev, device, torch.get_num_threads())
    return dev
