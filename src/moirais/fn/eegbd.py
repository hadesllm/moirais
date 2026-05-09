# moirais.fn — function file (hadesllm/moirais)
"""EEG frequency band decomposition plot."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_eeg_bands_fn(
    x: np.ndarray,
    fs: float = 256.0,
    title: str = "EEG Band Analysis",
) -> DescriptiveResult:
    """Plot EEG decomposed into delta, theta, alpha, beta, gamma bands.

    Parameters
    ----------
    x : array-like
        1-D EEG signal.
    fs : float
        Sampling frequency in Hz (default 256).
    title : str
        Plot title.

    Returns
    -------
    DescriptiveResult
        *value* is 5 (number of EEG bands); *extra* contains ``figure``.

    References
    ----------
    Niedermeyer, E. & da Silva, F. L. (2005). *Electroencephalography:
        Basic Principles, Clinical Applications, and Related Fields*.
        Lippincott Williams & Wilkins.
    """
    from moirais._bioplot import plot_eeg_bands

    x = np.asarray(x, dtype=float)
    fig = plot_eeg_bands(x, fs=fs, title=title)
    return DescriptiveResult(
        name="eeg_bands",
        value=5,
        extra={"figure": fig, "fs": fs},
    )


eegbd = plot_eeg_bands_fn


def cheatsheet() -> str:
    return "plot_eeg_bands_fn({}) -> EEG frequency band decomposition plot."
