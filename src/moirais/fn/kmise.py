# moirais.fn — function file (hadesllm/moirais)
"""MISE-optimal bandwidth for kernel density estimation."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kmise"]


def kmise(data: np.ndarray, *, kernel: str = "gaussian") -> dict:
    r"""
    MISE-optimal bandwidth assuming a Gaussian reference distribution.

    For a Gaussian kernel estimating a Gaussian density, the
    asymptotically optimal bandwidth minimizing the mean integrated
    squared error (MISE) is:

    .. math::

        h^* = \left(\frac{4\hat\sigma^5}{3n}\right)^{1/5}
        \approx 1.06\,\hat\sigma\,n^{-1/5}

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.
    kernel : str
        ``'gaussian'`` (default).

    Returns
    -------
    dict
        ``bw_opt``, ``mise_approx``, ``n``.

    References
    ----------
    Wand, M. P. & Jones, M. C. (1995). *Kernel Smoothing*. Chapman & Hall.
        Equation (3.9).
    """
    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    sigma = float(np.std(data, ddof=1))
    sigma = max(sigma, 1e-10)

    if kernel == "gaussian":
        bw_opt = (4.0 * sigma ** 5 / (3.0 * n)) ** 0.2
        rk = 1.0 / (2.0 * np.sqrt(np.pi))
        mise_approx = (5.0 / 4.0) * rk * n ** (-4.0 / 5) * (
            3.0 / (4.0 * sigma ** 5)
        ) ** (1.0 / 5)
    else:
        raise ValueError("Only 'gaussian' kernel is currently supported for MISE computation.")

    return RichResult(payload={"bw_opt": float(bw_opt), "mise_approx": float(mise_approx), "n": n})


def cheatsheet() -> str:
    return "kmise({data}) -> MISE-optimal bandwidth (Gaussian reference)."
