# morie.fn — function file (hadesllm/morie)
"""Ridge extraction from time-frequency representation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Every generation has a legend."


def ridge_extract(
    tfr: np.ndarray,
    fs: float = 1.0,
) -> DescriptiveResult:
    """Extract ridges (instantaneous frequency tracks) from a TF representation.

    Uses greedy peak-following across time columns.

    Parameters
    ----------
    tfr : 2-D array
        Time-frequency representation (freq x time), e.g. spectrogram power.
    fs : float
        Sampling frequency (default 1.0).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``ridge_indices`` (freq bin per time),
        ``ridge_frequencies`` (Hz), ``ridge_amplitude``.
    """
    tfr = np.asarray(tfr, dtype=float)
    if tfr.ndim != 2:
        raise ValueError("tfr must be 2-D (freq x time)")
    n_freq, n_time = tfr.shape
    ridge_idx = np.zeros(n_time, dtype=int)
    ridge_idx[0] = np.argmax(tfr[:, 0])
    for t in range(1, n_time):
        prev = ridge_idx[t - 1]
        lo = max(0, prev - 2)
        hi = min(n_freq, prev + 3)
        local = tfr[lo:hi, t]
        ridge_idx[t] = lo + np.argmax(local)
    freq_bins = np.linspace(0, fs / 2, n_freq)
    ridge_freq = freq_bins[ridge_idx]
    ridge_amp = np.array([tfr[ridge_idx[t], t] for t in range(n_time)])
    return DescriptiveResult(
        name="ridge_extract",
        value=float(np.mean(ridge_freq)),
        extra={
            "ridge_indices": ridge_idx,
            "ridge_frequencies": ridge_freq,
            "ridge_amplitude": ridge_amp,
        },
    )


ridgx = ridge_extract


def cheatsheet() -> str:
    return "ridge_extract({}) -> Ridge extraction from time-frequency representation."
