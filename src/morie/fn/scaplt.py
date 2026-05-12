# morie.fn -- function file (hadesllm/morie)
"""Wavelet scalogram plot visualization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_scalogram_fn(
    coeffs: np.ndarray,
    scales: np.ndarray,
    fs: float = 1.0,
    title: str = "Wavelet Scalogram",
) -> DescriptiveResult:
    """Plot a continuous wavelet transform scalogram.

    Parameters
    ----------
    coeffs : np.ndarray
        2-D wavelet coefficient array ``(n_scales, n_samples)``.
    scales : np.ndarray
        1-D array of wavelet scales.
    fs : float
        Sampling frequency in Hz.
    title : str
        Plot title.

    Returns
    -------
    DescriptiveResult
        *value* is number of scales; *extra* contains ``figure``.
    """
    from morie._bioplot import plot_scalogram

    coeffs = np.asarray(coeffs, dtype=float)
    scales = np.asarray(scales, dtype=float)
    fig = plot_scalogram(coeffs, scales, fs=fs, title=title)
    return DescriptiveResult(
        name="scalogram_plot",
        value=len(scales),
        extra={"figure": fig, "n_scales": len(scales)},
    )


scaplt = plot_scalogram_fn


def cheatsheet() -> str:
    return "plot_scalogram_fn({}) -> Wavelet scalogram plot visualization."
