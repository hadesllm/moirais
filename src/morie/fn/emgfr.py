# morie.fn — function file (hadesllm/morie)
"""EMG median and mean frequency.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 14.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['emgfr']

_QUOTE = "Frequency of the warrior. -- Ahsoka"


def emgfr(
    x: np.ndarray,
    fs: float = 1000.0,
    *,
    nperseg: int = 256,
) -> DescriptiveResult:
    """Compute median and mean frequency of an EMG signal.

    Parameters
    ----------
    x : array-like
        1-D EMG signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Segment length for PSD estimation.

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import welch

    x = np.asarray(x, dtype=float).ravel()
    seg = min(nperseg, len(x))
    f, psd = welch(x, fs=fs, nperseg=seg)

    total_power = np.sum(psd)
    if total_power < 1e-20:
        return DescriptiveResult(
            name="emgfr", value=0.0,
            extra={"median_freq": 0.0, "mean_freq": 0.0, "total_power": 0.0},
        )

    mean_freq = float(np.sum(f * psd) / total_power)

    cumpower = np.cumsum(psd)
    half = cumpower[-1] / 2.0
    idx = np.searchsorted(cumpower, half)
    idx = min(idx, len(f) - 1)
    median_freq = float(f[idx])

    return DescriptiveResult(
        name="emgfr",
        value=median_freq,
        extra={
            "median_freq": median_freq,
            "mean_freq": mean_freq,
            "total_power": float(total_power),
            "frequencies": f,
            "psd": psd,
        },
    )


def cheatsheet() -> str:
    return "emgfr({}) -> EMG median/mean frequency."
