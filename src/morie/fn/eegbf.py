# morie.fn -- function file (hadesllm/morie)
"""EEG band-pass filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def eeg_band_filter_fn(
    x: np.ndarray,
    fs: float = 256.0,
    band: str = "alpha",
) -> SignalResult:
    """Filter an EEG signal to a specific frequency band.

    Supported bands: delta (0.5--4 Hz), theta (4--8 Hz), alpha (8--13 Hz),
    beta (13--30 Hz), gamma (30--80 Hz).

    Parameters
    ----------
    x : array-like
        1-D EEG signal.
    fs : float
        Sampling frequency in Hz (default 256).
    band : str
        Band name (one of ``delta``, ``theta``, ``alpha``, ``beta``,
        ``gamma``).

    Returns
    -------
    SignalResult
        *filtered* is the band-passed signal; *extra* contains ``band``.

    References
    ----------
    Niedermeyer, E. & da Silva, F. L. (2005). *Electroencephalography*.
        Lippincott Williams & Wilkins.
    """
    from morie._bioplot import eeg_band_filter

    x = np.asarray(x, dtype=float)
    filtered = eeg_band_filter(x, fs=fs, band=band)
    return SignalResult(
        name="eeg_band_filter",
        filtered=filtered,
        fs=fs,
        n_samples=len(filtered),
        extra={"band": band},
    )


eegbf = eeg_band_filter_fn


def cheatsheet() -> str:
    return "eeg_band_filter_fn({}) -> EEG band-pass filter."
