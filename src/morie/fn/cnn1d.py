# morie.fn — function file (hadesllm/morie)
"""1D convolution forward pass (valid, cross-correlation convention)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["conv1d_forward"]


def conv1d_forward(x, w, b=0.0, stride: int = 1, padding: int = 0):
    """1D convolution / cross-correlation forward pass.

    .. math::

        y[i] = \\sum_{k=0}^{K-1} w[k]\\, x[i\\,s + k] + b

    using the cross-correlation convention standard in CNNs (no kernel
    flip), with ``'valid'`` padding by default.

    Parameters
    ----------
    x : array-like, shape ``(N,)``
        Input 1D signal.
    w : array-like, shape ``(K,)``
        Kernel.
    b : float
        Bias.
    stride : int
        Stride ``s``.
    padding : int
        Zero-padding added to both sides of ``x``.

    Returns
    -------
    result : RichResult
        Keys: ``y`` / ``estimate``, ``output_length``.

    References
    ----------
    Goodfellow, Bengio & Courville (2016) *Deep Learning*, Ch 9.
    """
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if padding > 0:
        x = np.pad(x, (padding, padding))
    K = w.shape[0]
    N = x.shape[0]
    if N < K:
        raise ValueError(f"Input length {N} < kernel length {K}.")
    out_len = (N - K) // stride + 1
    idx = np.arange(out_len) * stride
    windows = np.lib.stride_tricks.sliding_window_view(x, K)[idx]
    y = windows @ w + b
    return RichResult(
        title="Conv1D forward",
        summary_lines=[("input length", N), ("kernel length", K),
                       ("output length", out_len), ("stride", stride),
                       ("padding", padding)],
        payload={
            "y": y,
            "estimate": y,
            "output_length": int(out_len),
            "method": "Conv1D forward (cross-correlation)",
        },
    )


# CANONICAL TEST
# conv1d_forward([1,2,3,4,5], [1,0,-1]).y -> [1-3, 2-4, 3-5] = [-2, -2, -2]


def cheatsheet():
    return "cnn1d: 1D cross-correlation y[i] = sum w[k]*x[i*s+k]+b"
