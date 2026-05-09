"""Spectrogram plot visualization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_spectrogram_fn(
    Sxx: np.ndarray,
    fs: float = 1.0,
    t: np.ndarray | None = None,
    f: np.ndarray | None = None,
    title: str = "Spectrogram",
) -> DescriptiveResult:
    """Plot a time-frequency spectrogram.

    Parameters
    ----------
    Sxx : np.ndarray
        2-D spectrogram array ``(n_freq, n_time)``.
    fs : float
        Sampling frequency in Hz.
    t : np.ndarray, optional
        Time axis values.
    f : np.ndarray, optional
        Frequency axis values.
    title : str
        Plot title.

    Returns
    -------
    DescriptiveResult
        *value* is number of frequency bins; *extra* contains ``figure``.
    """
    from moirais._bioplot import plot_spectrogram

    Sxx = np.asarray(Sxx, dtype=float)
    fig = plot_spectrogram(Sxx, fs=fs, t=t, f=f, title=title)
    return DescriptiveResult(
        name="spectrogram_plot",
        value=Sxx.shape[0],
        extra={"figure": fig, "shape": Sxx.shape},
    )


sgmplt = plot_spectrogram_fn


def cheatsheet() -> str:
    return "plot_spectrogram_fn({}) -> Spectrogram plot visualization."
