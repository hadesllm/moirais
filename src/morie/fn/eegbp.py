# morie.fn — function file (hadesllm/morie)
"""EEG band power computation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def eeg_band_power_fn(
    x: np.ndarray,
    fs: float = 256.0,
) -> DescriptiveResult:
    """Compute absolute and relative power in standard EEG frequency bands.

    Bands: delta (0.5--4 Hz), theta (4--8 Hz), alpha (8--13 Hz),
    beta (13--30 Hz), gamma (30--80 Hz).

    Parameters
    ----------
    x : array-like
        1-D EEG signal.
    fs : float
        Sampling frequency in Hz (default 256).

    Returns
    -------
    DescriptiveResult
        *value* is total power; *extra* contains per-band absolute and
        relative powers.

    References
    ----------
    Welch, P. D. (1967). The use of fast Fourier transform for the
        estimation of power spectra. *IEEE Trans. Audio Electroacoustics*,
        15(2), 70--73.
    """
    from morie._bioplot import eeg_band_power

    x = np.asarray(x, dtype=float)
    powers = eeg_band_power(x, fs=fs)
    return DescriptiveResult(
        name="eeg_band_power",
        value=powers["total_power"],
        extra=powers,
    )


eegbp = eeg_band_power_fn


def cheatsheet() -> str:
    return "eeg_band_power_fn({}) -> EEG band power computation."
