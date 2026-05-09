# moirais.fn — function file (hadesllm/moirais)
"""Overlap-save fast convolution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You can't stop the change, any more than you can stop the suns from setting."


def overlap_save(x, h, block_size: int = 64, **kwargs) -> DescriptiveResult:
    """Compute linear convolution via the overlap-save method.

    Parameters
    ----------
    x : array-like
        Input signal.
    h : array-like
        Filter impulse response.
    block_size : int
        Block length (must be >= len(h)).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    M = len(h)
    N = block_size
    if N < M:
        N = M
    step = N - M + 1
    out_len = len(x) + M - 1
    total_needed = M - 1 + out_len
    n_blocks = int(np.ceil(out_len / step))
    pad_len = (M - 1) + n_blocks * step + (M - 1)
    x_padded = np.zeros(pad_len)
    x_padded[M - 1 : M - 1 + len(x)] = x
    y = np.zeros(out_len)
    H = np.fft.fft(h, n=N)
    pos = 0
    out_pos = 0
    while pos + N <= len(x_padded):
        block = x_padded[pos : pos + N]
        Y_block = np.fft.fft(block) * H
        seg = np.real(np.fft.ifft(Y_block))
        valid = seg[M - 1 :]
        end = min(out_pos + len(valid), out_len)
        y[out_pos:end] = valid[: end - out_pos]
        pos += step
        out_pos += step
    return DescriptiveResult(
        name="overlap_save",
        value=float(np.max(np.abs(y))),
        extra={"output": y, "block_size": N, "step": step},
    )


olsav = overlap_save


def cheatsheet() -> str:
    return "overlap_save({}) -> Overlap-save fast convolution."
