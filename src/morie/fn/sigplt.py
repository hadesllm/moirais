"""Generic signal plot visualization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_signal_fn(
    x: np.ndarray,
    fs: float = 1.0,
    title: str = "Signal",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
) -> DescriptiveResult:
    """Plot a single time-domain signal.

    Parameters
    ----------
    x : array-like
        1-D signal array.
    fs : float
        Sampling frequency in Hz.
    title, xlabel, ylabel : str
        Plot labels.

    Returns
    -------
    DescriptiveResult
        *value* is signal length; *extra* contains ``figure``.
    """
    from morie._bioplot import plot_signal

    x = np.asarray(x, dtype=float)
    fig = plot_signal(x, fs=fs, title=title, xlabel=xlabel, ylabel=ylabel)
    return DescriptiveResult(
        name="signal_plot",
        value=len(x),
        extra={"figure": fig, "fs": fs},
    )


sigplt = plot_signal_fn


def cheatsheet() -> str:
    return "plot_signal_fn({}) -> Generic signal plot visualization."
