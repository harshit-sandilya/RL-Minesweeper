"""
rl/common/utils.py

Shared low-level utilities used across all algorithms and agents.
No RL logic lives here — only infrastructure primitives.
"""

import random
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

# ── Reproducibility ───────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """
    Seed every RNG that could affect training.

    Covers Python random, NumPy, PyTorch CPU, and PyTorch CUDA.
    Also forces cuDNN into deterministic mode so conv results are
    reproducible across runs on the same hardware.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Device helpers ────────────────────────────────────────────────────────────


def get_device(device_str: str) -> torch.device:
    """
    Resolve a device string to a torch.device, with a safe CUDA fallback.

    If "cuda" is requested but no GPU is available, issues a warning and
    returns CPU instead of crashing at runtime.

    Args:
        device_str: "cpu", "cuda", or a specific device like "cuda:0".

    Returns:
        Resolved torch.device.
    """
    if "cuda" in device_str and not torch.cuda.is_available():
        print(
            f"[utils] WARNING: device='{device_str}' requested but CUDA is not available. Falling back to CPU."
        )
        return torch.device("cpu")
    return torch.device(device_str)


# ── Rolling statistics ────────────────────────────────────────────────────────


class RollingMean:
    """
    Compute the mean of the last `window` values without NumPy or SciPy.

    Used by Logger to maintain the rolling 100-episode win_rate that
    TensorBoard logs under train/win_rate.

    Args:
        window: Number of most-recent values to include in the mean.

    Example:
        >>> rm = RollingMean(window=100)
        >>> rm.push(1.0)
        >>> rm.push(0.0)
        >>> rm.mean()          # 0.5
    """

    def __init__(self, window: int) -> None:
        self._buf: deque = deque(maxlen=window)

    def push(self, value: float) -> None:
        """Append a new value; oldest is evicted once the window is full."""
        self._buf.append(float(value))

    def mean(self) -> float:
        """
        Return the current rolling mean.

        Returns 0.0 when the buffer is empty so callers never need a
        special-case check at the start of training.
        """
        if not self._buf:
            return 0.0
        return sum(self._buf) / len(self._buf)

    def __len__(self) -> int:
        return len(self._buf)

    def is_full(self) -> bool:
        """True once the buffer has seen at least `window` values."""
        return len(self._buf) == self._buf.maxlen


# ── Timing ────────────────────────────────────────────────────────────────────


class Timer:
    """
    Context manager that measures elapsed wall-clock time in milliseconds.

    Used by the agent to populate perf/inference_ms and
    perf/wall_time_per_ep in TensorBoard.

    Example:
        >>> with Timer() as t:
        ...     _ = model(state)
        >>> print(t.elapsed_ms)   # e.g. 0.37
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        assert self._start is not None
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000.0
