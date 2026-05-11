# morie.fn — function file (hadesllm/morie)
"""Noise floor estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Character is destiny. — Heraclitus"


def noise_floor(x, method: str = "median", **kwargs) -> DescriptiveResult:
    """Estimate the noise floor of a signal spectrum.

    Parameters
    ----------
    x : array-like
        Input signal (time domain).
    method : str
        Estimation method: ``"median"`` or ``"percentile"``.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    X_mag = np.abs(np.fft.rfft(x))
    X_pow = X_mag**2
    if method == "median":
        floor_val = float(np.median(X_pow))
    elif method == "percentile":
        floor_val = float(np.percentile(X_pow, 10))
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'median' or 'percentile'.")
    floor_db = 10.0 * np.log10(max(floor_val, 1e-30))
    return DescriptiveResult(
        name="noise_floor",
        value=floor_db,
        extra={"floor_power": floor_val, "floor_db": floor_db, "method": method},
    )


nflor = noise_floor


def cheatsheet() -> str:
    return "noise_floor({}) -> Noise floor estimation."
