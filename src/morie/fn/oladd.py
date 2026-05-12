# morie.fn -- function file (hadesllm/morie)
"""Overlap-add fast convolution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It's a trap!"


def overlap_add(x, h, block_size: int = 64, **kwargs) -> DescriptiveResult:
    """Compute linear convolution via the overlap-add method.

    Parameters
    ----------
    x : array-like
        Input signal.
    h : array-like
        Filter impulse response.
    block_size : int
        Block length for segmentation.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    M = len(h)
    L = block_size
    N_fft = L + M - 1
    out_len = len(x) + M - 1
    y = np.zeros(out_len)
    H = np.fft.fft(h, n=N_fft)
    n_blocks = int(np.ceil(len(x) / L))
    for i in range(n_blocks):
        block = x[i * L : (i + 1) * L]
        X_block = np.fft.fft(block, n=N_fft)
        seg = np.real(np.fft.ifft(X_block * H))
        start = i * L
        end = min(start + N_fft, out_len)
        y[start:end] += seg[: end - start]
    return DescriptiveResult(
        name="overlap_add",
        value=float(np.max(np.abs(y))),
        extra={"output": y, "block_size": L, "n_blocks": n_blocks},
    )


oladd = overlap_add


def cheatsheet() -> str:
    return "overlap_add({}) -> Overlap-add fast convolution."
