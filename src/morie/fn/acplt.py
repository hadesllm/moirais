# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Autocorrelation plot data."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def autocorrelation_data(
    samples: Union[list, np.ndarray],
    *,
    max_lag: int = 50,
) -> dict[str, Any]:
    """
    Compute autocorrelation function (ACF) for MCMC diagnostics.

    :param samples: MCMC samples (1-D array).
    :param max_lag: Maximum lag to compute.
    :return: Dictionary with lags, acf values, first lag below 0.05.
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    max_lag = min(max_lag, n - 1)

    mean_x = np.mean(x)
    c0 = float(np.sum((x - mean_x) ** 2))
    if c0 == 0:
        return RichResult(payload={"lags": list(range(max_lag + 1)), "acf": [1.0] + [0.0] * max_lag, "first_below_threshold": 1})

    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            c_lag = float(np.sum((x[:-lag] - mean_x) * (x[lag:] - mean_x)))
            acf_vals.append(c_lag / c0)

    first_below = max_lag + 1
    for i, v in enumerate(acf_vals):
        if i > 0 and abs(v) < 0.05:
            first_below = i
            break

    return {
        "lags": list(range(max_lag + 1)),
        "acf": acf_vals,
        "first_below_threshold": first_below,
        "n": n,
    }


acplt = autocorrelation_data


def cheatsheet() -> str:
    return "autocorrelation_data({}) -> Autocorrelation plot data."
