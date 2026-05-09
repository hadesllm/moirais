"""Power spectrum plot visualization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_spectrum_fn(
    psd: np.ndarray,
    freqs: np.ndarray,
    log_scale: bool = True,
    title: str = "Power Spectrum",
) -> DescriptiveResult:
    """Plot a power spectral density curve.

    Parameters
    ----------
    psd : array-like
        Power spectral density values.
    freqs : array-like
        Corresponding frequency values in Hz.
    log_scale : bool
        Use logarithmic y-axis (default True).
    title : str
        Plot title.

    Returns
    -------
    DescriptiveResult
        *value* is number of frequency bins; *extra* contains ``figure``.
    """
    from moirais._bioplot import plot_spectrum

    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    fig = plot_spectrum(psd, freqs, log_scale=log_scale, title=title)
    return DescriptiveResult(
        name="spectrum_plot",
        value=len(psd),
        extra={"figure": fig, "log_scale": log_scale},
    )


spcplt = plot_spectrum_fn


def cheatsheet() -> str:
    return "plot_spectrum_fn({}) -> Power spectrum plot visualization."
