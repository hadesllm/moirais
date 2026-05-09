# moirais.fn — function file (hadesllm/moirais)
"""EMG RMS envelope.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 14.
"""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

__all__ = ['emgrm']

_QUOTE = "Strength in every muscle. --"


def emgrm(
    x: np.ndarray,
    fs: float = 1000.0,
    *,
    window_ms: float = 50.0,
) -> SignalResult:
    """Compute the RMS envelope of an EMG signal.

    Parameters
    ----------
    x : array-like
        1-D EMG signal.
    fs : float
        Sampling frequency in Hz.
    window_ms : float
        RMS window duration in milliseconds.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float).ravel()
    win = max(1, int(window_ms * fs / 1000.0))
    n = len(x)

    rms = np.zeros(n)
    x2 = x ** 2
    cumsum = np.cumsum(x2)
    cumsum = np.insert(cumsum, 0, 0)

    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, i + win // 2 + 1)
        rms[i] = np.sqrt((cumsum[hi] - cumsum[lo]) / (hi - lo))

    return SignalResult(
        name="emgrm",
        filtered=rms,
        fs=fs,
        n_samples=n,
        extra={"window_ms": window_ms, "window_samples": win},
    )


def cheatsheet() -> str:
    return "emgrm({}) -> EMG RMS envelope."
