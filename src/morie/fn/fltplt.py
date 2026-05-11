# morie.fn — function file (hadesllm/morie)
"""Filter input/output comparison plot."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_filter_io_fn(
    x_in: np.ndarray,
    x_out: np.ndarray,
    fs: float = 1.0,
    title: str = "Filter Input/Output",
) -> DescriptiveResult:
    """Plot filter input and output signals side by side.

    Parameters
    ----------
    x_in : array-like
        Input signal.
    x_out : array-like
        Output (filtered) signal.
    fs : float
        Sampling frequency in Hz.
    title : str
        Plot title.

    Returns
    -------
    DescriptiveResult
        *value* is input signal length; *extra* contains ``figure``.
    """
    from morie._bioplot import plot_filter_io

    x_in = np.asarray(x_in, dtype=float)
    x_out = np.asarray(x_out, dtype=float)
    fig = plot_filter_io(x_in, x_out, fs=fs, title=title)
    return DescriptiveResult(
        name="filter_io_plot",
        value=len(x_in),
        extra={"figure": fig, "fs": fs},
    )


fltplt = plot_filter_io_fn


def cheatsheet() -> str:
    return "plot_filter_io_fn({}) -> Filter input/output comparison plot."
