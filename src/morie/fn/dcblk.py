# morie.fn -- function file (hadesllm/morie)
"""DC blocker (remove zero-frequency component).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
"""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

__all__ = ['dcblk']
def dcblk(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    method: str = "mean",
    alpha: float = 0.995,
) -> SignalResult:
    """Remove DC offset from a signal.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    method : str
        ``'mean'`` subtracts the sample mean; ``'iir'`` uses a
        single-pole high-pass IIR: y[n] = x[n] - x[n-1] + alpha*y[n-1].
    alpha : float
        Pole for IIR method (default 0.995).

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float).ravel()
    dc_value = float(np.mean(x))

    if method == "iir":
        y = np.empty_like(x)
        y[0] = 0.0
        for i in range(1, len(x)):
            y[i] = x[i] - x[i - 1] + alpha * y[i - 1]
    else:
        y = x - dc_value

    return SignalResult(
        name="dcblk",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"dc_removed": dc_value, "method": method},
    )


def cheatsheet() -> str:
    return "dcblk({}) -> DC blocker."
