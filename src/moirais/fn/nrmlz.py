# moirais.fn — function file (hadesllm/moirais)
"""Signal normalization."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "We are what they grow beyond."


def normalize_signal(x, method: str = "zscore") -> SignalResult:
    """Normalize signal *x* using the specified method.

    Parameters
    ----------
    x : array-like
        Input signal.
    method : str
        'zscore' (mean=0, std=1), 'minmax' ([0,1]), or 'energy' (unit energy).

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    if method == "zscore":
        s = np.std(x)
        y = (x - np.mean(x)) / s if s > 0 else x - np.mean(x)
    elif method == "minmax":
        mn, mx = np.min(x), np.max(x)
        rng = mx - mn
        y = (x - mn) / rng if rng > 0 else np.zeros_like(x)
    elif method == "energy":
        e = np.sqrt(np.sum(x**2))
        y = x / e if e > 0 else x.copy()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore', 'minmax', or 'energy'.")
    return SignalResult(
        name="normalize_signal",
        filtered=y,
        fs=0.0,
        n_samples=len(x),
        extra={"method": method},
    )


nrmlz = normalize_signal


def cheatsheet() -> str:
    return "normalize_signal({}) -> Signal normalization."
