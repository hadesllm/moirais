"""Time-frequency distribution plot."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_tfd_fn(
    tfd: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    title: str = "Time-Frequency Distribution",
) -> DescriptiveResult:
    """Plot a general time-frequency distribution (Wigner-Ville, etc.).

    Parameters
    ----------
    tfd : np.ndarray
        2-D TFD array ``(n_freq, n_time)``.
    t : np.ndarray
        Time axis values.
    f : np.ndarray
        Frequency axis values.
    title : str
        Plot title.

    Returns
    -------
    DescriptiveResult
        *value* is number of frequency bins; *extra* contains ``figure``.

    References
    ----------
    Cohen, L. (1995). *Time-Frequency Analysis*. Prentice Hall.
    """
    from moirais._bioplot import plot_tfd

    tfd = np.asarray(tfd, dtype=float)
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    fig = plot_tfd(tfd, t, f, title=title)
    return DescriptiveResult(
        name="tfd_plot",
        value=tfd.shape[0],
        extra={"figure": fig, "shape": tfd.shape},
    )


tfplt = plot_tfd_fn


def cheatsheet() -> str:
    return "plot_tfd_fn({}) -> Time-frequency distribution plot."
