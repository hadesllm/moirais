# morie.fn -- function file (hadesllm/morie)
"""Hilbert-Huang Transform (EMD + Hilbert spectrum)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def _sift(x: np.ndarray, max_iter: int = 300, tol: float = 0.05) -> np.ndarray:
    """Extract one IMF via sifting."""
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


def hilbert_huang(
    x: np.ndarray,
    fs: float = 1.0,
    max_imfs: int = 10,
) -> DescriptiveResult:
    """Hilbert-Huang Transform: Empirical Mode Decomposition + Hilbert spectrum.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency (default 1.0).
    max_imfs : int
        Maximum number of IMFs to extract (default 10).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``imfs``, ``residue``, ``inst_freqs``, ``inst_amps``.
    """
    x = np.asarray(x, dtype=float).ravel()
    residue = x.copy()
    imfs = []
    for _ in range(max_imfs):
        imf = _sift(residue)
        if np.max(np.abs(imf)) < 1e-10:
            break
        imfs.append(imf)
        residue = residue - imf
        t = np.arange(len(residue))
        max_idx = np.where((residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:]))[0] + 1
        min_idx = np.where((residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
    inst_freqs = []
    inst_amps = []
    for imf in imfs:
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.gradient(phase) * fs / (2 * np.pi)
        inst_freqs.append(freq)
        inst_amps.append(amp)
    return DescriptiveResult(
        name="hilbert_huang",
        value=float(len(imfs)),
        extra={
            "imfs": imfs,
            "residue": residue,
            "inst_freqs": inst_freqs,
            "inst_amps": inst_amps,
        },
    )


hhtfn = hilbert_huang


def cheatsheet() -> str:
    return "_sift({}) -> Hilbert-Huang Transform (EMD + Hilbert spectrum)."
