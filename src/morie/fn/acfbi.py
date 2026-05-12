# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Biased autocorrelation function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Always pass on what you have learned."


def acf_biased(x, maxlag: int | None = None, **kwargs) -> DescriptiveResult:
    r"""Compute the biased autocorrelation function.

    .. math::

        R_{xx}(m) = \\frac{1}{N} \\sum_{n=0}^{N-1-|m|} x(n) \\cdot x(n + |m|)

    Parameters
    ----------
    x : array-like
        Input signal.
    maxlag : int or None
        Maximum lag. Default: N-1.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    N = len(x)
    if maxlag is None:
        maxlag = N - 1
    maxlag = min(maxlag, N - 1)
    lags = np.arange(0, maxlag + 1)
    acf = np.array([np.sum(x[: N - m] * x[m:]) / N for m in lags])
    return DescriptiveResult(
        name="acf_biased",
        value=float(acf[0]),
        extra={"acf": acf, "lags": lags, "maxlag": maxlag, "N": N},
    )


acfbi = acf_biased


def cheatsheet() -> str:
    return "acf_biased({}) -> Biased autocorrelation function."
