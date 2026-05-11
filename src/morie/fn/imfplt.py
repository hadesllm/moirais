# morie.fn — function file (hadesllm/morie)
"""IMF decomposition stack plot."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_imf_stack_fn(
    imfs: list[np.ndarray],
    fs: float = 1.0,
    title: str = "IMF Decomposition",
) -> DescriptiveResult:
    """Plot intrinsic mode functions (IMFs) in a stacked layout.

    Parameters
    ----------
    imfs : list[np.ndarray]
        List of 1-D IMF arrays from empirical mode decomposition.
    fs : float
        Sampling frequency in Hz.
    title : str
        Plot title.

    Returns
    -------
    DescriptiveResult
        *value* is number of IMFs; *extra* contains ``figure``.

    References
    ----------
    Huang, N. E. et al. (1998). The empirical mode decomposition and the
        Hilbert spectrum for nonlinear and non-stationary time series
        analysis. *Proc. R. Soc. London A*, 454, 903--995.
    """
    from morie._bioplot import plot_imf_stack

    n_imfs = len(imfs)
    fig = plot_imf_stack(imfs, fs=fs, title=title)
    return DescriptiveResult(
        name="imf_plot",
        value=n_imfs,
        extra={"figure": fig, "n_imfs": n_imfs},
    )


imfplt = plot_imf_stack_fn


def cheatsheet() -> str:
    return "plot_imf_stack_fn({}) -> IMF decomposition stack plot."
