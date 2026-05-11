"""Weighted spectral slope distance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Every choice you have made has led you to this moment."


def wss_distance(S1, S2, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute weighted spectral slope distance.

    Measures the perceptually weighted difference in spectral slopes
    between two spectra, commonly used in speech quality assessment.

    Parameters
    ----------
    S1 : array-like
        Power spectrum 1.
    S2 : array-like
        Power spectrum 2.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    S1 = np.maximum(S1, 1e-30)
    S2 = np.maximum(S2, 1e-30)
    S1_db = 10.0 * np.log10(S1)
    S2_db = 10.0 * np.log10(S2)
    slope1 = np.diff(S1_db)
    slope2 = np.diff(S2_db)
    weights = S1_db[:-1] / np.max(S1_db[:-1]) if np.max(S1_db[:-1]) > 0 else np.ones_like(slope1)
    weights = np.maximum(weights, 0.0)
    wss = float(np.sqrt(np.mean(weights * (slope1 - slope2) ** 2)))
    return DescriptiveResult(
        name="wss_distance",
        value=wss,
        extra={"wss": wss, "fs": fs},
    )


wssds = wss_distance


def cheatsheet() -> str:
    return "wss_distance({}) -> Weighted spectral slope distance."
