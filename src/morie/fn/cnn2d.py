# morie.fn — function file (hadesllm/morie)
"""2D convolution forward pass (valid, cross-correlation convention)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["conv2d_forward"]


def conv2d_forward(x, w, b=0.0, stride: int = 1, padding: int = 0):
    r"""2D convolution / cross-correlation forward pass on one channel.

    .. math::

        y[i,j] = \\sum_{m=0}^{K_h-1}\\sum_{n=0}^{K_w-1}
                 w[m,n]\\, x[i\\,s+m,\\, j\\,s+n] + b

    Parameters
    ----------
    x : array-like, shape ``(H, W)``
        Input feature map.
    w : array-like, shape ``(K_h, K_w)``
        Kernel.
    b : float
        Bias.
    stride : int
        Stride.
    padding : int
        Zero-padding on each side.

    Returns
    -------
    result : RichResult
        Keys: ``y`` / ``estimate``, ``output_shape``.

    References
    ----------
    Goodfellow, Bengio & Courville (2016) *Deep Learning*, Ch 9.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("conv2d_forward expects 2D x and w.")
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)))
    H, W = x.shape
    Kh, Kw = w.shape
    if H < Kh or W < Kw:
        raise ValueError(f"Input {(H, W)} smaller than kernel {(Kh, Kw)}.")
    out_h = (H - Kh) // stride + 1
    out_w = (W - Kw) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (Kh, Kw))
    windows = windows[::stride, ::stride]
    y = np.einsum("ijkl,kl->ij", windows, w) + b
    return RichResult(
        title="Conv2D forward",
        summary_lines=[("input shape", x.shape), ("kernel shape", (Kh, Kw)),
                       ("output shape", (out_h, out_w)), ("stride", stride),
                       ("padding", padding)],
        payload={
            "y": y,
            "estimate": y,
            "output_shape": (int(out_h), int(out_w)),
            "method": "Conv2D forward (cross-correlation)",
        },
    )


# CANONICAL TEST
# x=[[1,2,3],[4,5,6],[7,8,9]], w=[[1,0],[0,-1]] ->
#   y[0,0] = 1 - 5 = -4
#   y[0,1] = 2 - 6 = -4
#   y[1,0] = 4 - 8 = -4
#   y[1,1] = 5 - 9 = -4


def cheatsheet():
    return "cnn2d: 2D cross-correlation y[i,j] = sum_mn w[m,n]*x[i+m,j+n]"
