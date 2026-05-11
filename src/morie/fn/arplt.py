# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""AR model pole-zero and spectrum plot."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_ar_poles_fn(
    coeffs: np.ndarray,
    fs: float = 1.0,
    title: str = "AR Model Poles",
) -> DescriptiveResult:
    """Plot AR model poles on the unit circle and the AR power spectrum.

    Parameters
    ----------
    coeffs : array-like
        AR model coefficients (excluding the leading 1).
    fs : float
        Sampling frequency in Hz.
    title : str
        Plot title.

    Returns
    -------
    DescriptiveResult
        *value* is AR model order; *extra* contains ``figure``.

    References
    ----------
    Kay, S. M. (1988). *Modern Spectral Estimation: Theory and
        Application*. Prentice Hall.
    """
    from morie._bioplot import plot_ar_poles

    coeffs = np.asarray(coeffs, dtype=float)
    fig = plot_ar_poles(coeffs, fs=fs, title=title)
    return DescriptiveResult(
        name="ar_poles_plot",
        value=len(coeffs),
        extra={"figure": fig, "order": len(coeffs)},
    )


arplt = plot_ar_poles_fn


def cheatsheet() -> str:
    return "plot_ar_poles_fn({}) -> AR model pole-zero and spectrum plot."
