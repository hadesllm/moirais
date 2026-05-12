# morie.fn -- function file (hadesllm/morie)
"""Extract intrinsic mode functions via sifting."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def _sift(x: np.ndarray, max_iter: int = 300, tol: float = 0.05) -> np.ndarray:
    h = x.copy()
    for _ in range(max_iter):
        t = np.arange(len(h))
        max_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
        min_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
        upper = CubicSpline(max_idx, h[max_idx], extrapolate=True)(t)
        lower = CubicSpline(min_idx, h[min_idx], extrapolate=True)(t)
        mean_env = (upper + lower) / 2
        prev = h.copy()
        h = h - mean_env
        sd = np.sum((prev - h) ** 2) / (np.sum(prev**2) + 1e-12)
        if sd < tol:
            break
    return h


def imf_extract(x, n_imfs: int = 5) -> DescriptiveResult:
    """Extract intrinsic mode functions via sifting.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    n_imfs : int
        Maximum number of IMFs to extract.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    residue = x.copy()
    imfs = []
    for _ in range(n_imfs):
        imf = _sift(residue)
        if np.max(np.abs(imf)) < 1e-10:
            break
        imfs.append(imf)
        residue = residue - imf
        max_idx = np.where((residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:]))[0] + 1
        min_idx = np.where((residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
    return DescriptiveResult(
        name="imf_extract",
        value=float(len(imfs)),
        extra={"imfs": imfs, "residue": residue, "n_imfs_requested": n_imfs},
    )


imfex = imf_extract


def cheatsheet() -> str:
    return "_sift({}) -> Extract intrinsic mode functions via sifting."
