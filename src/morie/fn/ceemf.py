# morie.fn — function file (hadesllm/morie)
"""Complete Ensemble EMD (CEEMD) decomposition."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ._containers import DescriptiveResult

_QUOTE = "Power, unlimited power!"


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


def _emd(x, max_imfs=5):
    residue = x.copy()
    imfs = []
    for _ in range(max_imfs):
        imf = _sift(residue)
        if np.max(np.abs(imf)) < 1e-10:
            break
        imfs.append(imf)
        residue = residue - imf
        max_idx = np.where((residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:]))[0] + 1
        min_idx = np.where((residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
    return imfs


def ceemd_decompose(
    x,
    n_imfs: int = 5,
    noise_std: float = 0.2,
    n_trials: int = 50,
) -> DescriptiveResult:
    """Complete Ensemble EMD decomposition.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    n_imfs : int
        Max number of IMFs.
    noise_std : float
        Noise amplitude as fraction of signal std.
    n_trials : int
        Number of noise-added trials.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(0)
    sigma = noise_std * np.std(x)
    accum = None
    count = 0
    for _ in range(n_trials):
        noise = rng.standard_normal(len(x)) * sigma
        imfs_pos = _emd(x + noise, max_imfs=n_imfs)
        imfs_neg = _emd(x - noise, max_imfs=n_imfs)
        for trial_imfs in (imfs_pos, imfs_neg):
            n_found = len(trial_imfs)
            if accum is None:
                accum = [np.zeros_like(x) for _ in range(n_found)]
            for i in range(min(n_found, len(accum))):
                accum[i] += trial_imfs[i][: len(x)]
            count += 1
    if accum is None:
        accum = []
    else:
        accum = [a / count for a in accum]
    return DescriptiveResult(
        name="ceemd_decompose",
        value=float(len(accum)),
        extra={"imfs": accum, "n_trials": n_trials, "noise_std": noise_std},
    )


ceemf = ceemd_decompose


def cheatsheet() -> str:
    return "_sift({}) -> Complete Ensemble EMD (CEEMD) decomposition."
