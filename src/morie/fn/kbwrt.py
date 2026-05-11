# morie.fn — function file (hadesllm/morie)
"""Bandwidth selection: Silverman's rule of thumb."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kbwrt"]


def kbwrt(data: np.ndarray, *, kernel: str = "gaussian") -> dict:
    r"""
    Silverman's rule-of-thumb bandwidth selector.

    For the Gaussian kernel:

    .. math::

        h = 0.9 \min\!\left(\hat\sigma,\;\frac{\mathrm{IQR}}{1.349}\right) n^{-1/5}

    For other kernels, a canonical scaling constant is applied.

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.
    kernel : str
        ``'gaussian'``, ``'epanechnikov'``, ``'biweight'``, ``'triweight'``.

    Returns
    -------
    dict
        ``bw``, ``sigma``, ``iqr``, ``n``, ``kernel``.

    References
    ----------
    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
        Analysis*. Chapman & Hall. Rule (3.31).
    """
    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    sigma = float(np.std(data, ddof=1))
    iqr = float(np.subtract(*np.percentile(data, [75, 25])))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    s = max(s, 1e-10)

    h_gauss = 0.9 * s * n ** (-0.2)

    canon = {
        "gaussian": 1.0,
        "epanechnikov": 2.34,
        "biweight": 2.78,
        "triweight": 3.15,
    }
    if kernel not in canon:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from {list(canon)}.")

    bw = h_gauss * (canon[kernel] / canon["gaussian"])

    return RichResult(payload={"bw": float(bw), "sigma": sigma, "iqr": iqr, "n": n, "kernel": kernel})


def cheatsheet() -> str:
    return "kbwrt({data}) -> Silverman's rule-of-thumb bandwidth."
