"""Total harmonic distortion."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful, the mind of a child is."


def thd_compute(x, fs: float = 1.0, n_harmonics: int = 5, **kwargs) -> DescriptiveResult:
    """Compute total harmonic distortion (THD).

    .. math::

        \\text{THD} = \\frac{\\sqrt{\\sum_{k=2}^{K} P_k}}{\\sqrt{P_1}}

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    n_harmonics : int
        Number of harmonics to include.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    fund_idx = np.argmax(mag[1:]) + 1
    fund_power = mag[fund_idx] ** 2
    harm_power = 0.0
    harmonics = []
    for k in range(2, n_harmonics + 1):
        h_idx = fund_idx * k
        if h_idx < len(mag):
            harm_power += mag[h_idx] ** 2
            harmonics.append((k, int(h_idx), float(mag[h_idx] ** 2)))
    if fund_power <= 0:
        thd_val = float("inf")
    else:
        thd_val = float(np.sqrt(harm_power / fund_power))
    thd_db = 20.0 * np.log10(max(thd_val, 1e-30))
    return DescriptiveResult(
        name="thd",
        value=thd_val,
        extra={
            "thd": thd_val,
            "thd_db": float(thd_db),
            "fund_bin": int(fund_idx),
            "fund_power": float(fund_power),
            "harmonics": harmonics,
            "n_harmonics": n_harmonics,
        },
    )


thd = thd_compute


def cheatsheet() -> str:
    return "thd_compute({}) -> Total harmonic distortion."
