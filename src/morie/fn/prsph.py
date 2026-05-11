# morie.fn — function file (hadesllm/morie)
"""Fidelity metric (SSIM). 'Tell me, is the woman worth it?' -- Persephone"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ssim(
    x: np.ndarray,
    y: np.ndarray,
    *,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
    win_size: int = 7,
) -> DescriptiveResult:
    """Structural Similarity Index (SSIM) between two signals or images.

    For 1-D signals, uses a sliding window. For 2-D images, computes
    the mean SSIM across all local windows.

    Parameters
    ----------
    x, y : ndarray
        Two arrays of the same shape (1-D or 2-D), values in [0, 1].
    C1, C2 : float
        Stability constants (default assumes dynamic range of 1).
    win_size : int
        Local window size (must be odd).

    Returns
    -------
    DescriptiveResult
        ``value`` is the mean SSIM in [-1, 1].

    References
    ----------
    Wang, Z. et al. (2004). Image quality assessment: from error visibility
    to structural similarity. IEEE TIP, 13(4), 600-612.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if win_size % 2 == 0:
        raise ValueError("win_size must be odd")

    if x.ndim == 1:
        n = len(x)
        if n < win_size:
            raise ValueError("Signal shorter than window")
        ssim_vals = []
        half = win_size // 2
        for i in range(half, n - half):
            wx = x[i - half : i + half + 1]
            wy = y[i - half : i + half + 1]
            mx, my = wx.mean(), wy.mean()
            sx2 = wx.var()
            sy2 = wy.var()
            sxy = float(np.mean((wx - mx) * (wy - my)))
            num = (2 * mx * my + C1) * (2 * sxy + C2)
            den = (mx**2 + my**2 + C1) * (sx2 + sy2 + C2)
            ssim_vals.append(num / den)
        mssim = float(np.mean(ssim_vals))
    elif x.ndim == 2:
        rows, cols = x.shape
        half = win_size // 2
        if rows < win_size or cols < win_size:
            raise ValueError("Image smaller than window")
        ssim_vals = []
        for i in range(half, rows - half):
            for j in range(half, cols - half):
                wx = x[i - half : i + half + 1, j - half : j + half + 1]
                wy = y[i - half : i + half + 1, j - half : j + half + 1]
                mx, my = wx.mean(), wy.mean()
                sx2 = wx.var()
                sy2 = wy.var()
                sxy = float(np.mean((wx - mx) * (wy - my)))
                num = (2 * mx * my + C1) * (2 * sxy + C2)
                den = (mx**2 + my**2 + C1) * (sx2 + sy2 + C2)
                ssim_vals.append(num / den)
        mssim = float(np.mean(ssim_vals))
    else:
        raise ValueError("x must be 1-D or 2-D")

    return DescriptiveResult(
        name="SSIM",
        value=mssim,
        extra={"win_size": win_size, "shape": list(x.shape), "n_windows": len(ssim_vals)},
    )


prsph = ssim


def cheatsheet() -> str:
    return "ssim({}) -> Fidelity metric (SSIM). 'Tell me, is the woman worth it?' --"
