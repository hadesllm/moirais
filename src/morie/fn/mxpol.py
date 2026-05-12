# morie.fn — function file (hadesllm/morie)
"""Max pooling forward pass (2D)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["maxpool_forward"]


def maxpool_forward(x, kernel_size: int = 2, stride=None):
    """2D max-pooling forward pass on one channel.

    .. math::

        y[i,j] = \\max_{0\\le m,n < k}
                 x[i\\,s+m,\\; j\\,s+n]

    Parameters
    ----------
    x : array-like, shape ``(H, W)``
        Input feature map.
    kernel_size : int
        Pool window size :math:`k`.
    stride : int or None
        Stride. Defaults to ``kernel_size`` (non-overlapping).

    Returns
    -------
    result : RichResult
        Keys: ``y`` / ``estimate``, ``argmax`` (flat index per window),
        ``output_shape``.

    References
    ----------
    Goodfellow, Bengio & Courville (2016) *Deep Learning*, Ch 9.3.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("maxpool_forward expects 2D x.")
    k = int(kernel_size)
    s = int(stride) if stride is not None else k
    H, W = x.shape
    if H < k or W < k:
        raise ValueError(f"Input {(H, W)} smaller than kernel {k}.")
    out_h = (H - k) // s + 1
    out_w = (W - k) // s + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (k, k))[::s, ::s]
    flat = windows.reshape(out_h, out_w, k * k)
    y = flat.max(axis=-1)
    argmax = flat.argmax(axis=-1)
    return RichResult(
        title="MaxPool2D forward",
        summary_lines=[("kernel", k), ("stride", s),
                       ("input shape", (H, W)),
                       ("output shape", (out_h, out_w))],
        payload={
            "y": y,
            "estimate": y,
            "argmax": argmax,
            "output_shape": (int(out_h), int(out_w)),
            "method": "MaxPool2D forward",
        },
    )


# CANONICAL TEST
# x=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], k=2, s=2
# y = [[6,8],[14,16]]


def cheatsheet():
    return "mxpol: MaxPool2D y[i,j] = max over k x k window at stride s"
