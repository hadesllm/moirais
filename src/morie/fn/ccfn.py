# morie.fn -- function file (hadesllm/morie)
"""Normalized cross-correlation function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def ccf_normalized(x, y, maxlag: int | None = None, **kwargs) -> DescriptiveResult:
    """Compute the normalized cross-correlation function.

    The result is normalized by the geometric mean of the zero-lag
    autocorrelations of *x* and *y*, yielding values in [-1, 1].

    Parameters
    ----------
    x, y : array-like
        Input signals.
    maxlag : int or None
        Maximum lag. Default: N-1.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    N = min(len(x), len(y))
    x = x[:N]
    y = y[:N]
    if maxlag is None:
        maxlag = N - 1
    maxlag = min(maxlag, N - 1)
    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if norm == 0:
        return DescriptiveResult(name="ccf_normalized", value=0.0, extra={"ccf": np.zeros(maxlag + 1)})
    lags = np.arange(0, maxlag + 1)
    ccf = np.array([np.sum(x[: N - m] * y[m:]) / norm for m in lags])
    return DescriptiveResult(
        name="ccf_normalized",
        value=float(np.max(np.abs(ccf))),
        extra={"ccf": ccf, "lags": lags, "maxlag": maxlag},
    )


ccfn = ccf_normalized


def cheatsheet() -> str:
    return "ccf_normalized({}) -> Normalized cross-correlation function."
