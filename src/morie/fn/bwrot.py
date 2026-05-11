# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bandwidth selection via Silverman rule-of-thumb."""

from __future__ import annotations

import numpy as np


def bwrot(
    x: np.ndarray,
    *,
    kernel: str = "gaussian",
    method: str = "silverman",
) -> dict:
    r"""
    Bandwidth selection via rule-of-thumb (Silverman / Scott).

    **Silverman (1986)** normal reference rule:

    .. math::

        h = 1.06 \cdot \min\!\left(\hat\sigma,\,
        \frac{\text{IQR}}{1.34}\right) n^{-1/5}

    **Scott (1992)** variant:

    .. math::

        h = 1.059 \cdot \hat\sigma \cdot n^{-1/5}

    Parameters
    ----------
    x : np.ndarray
        1-d array of observed values (n,).
    kernel : str
        Kernel: ``'gaussian'``, ``'epanechnikov'``, or ``'uniform'``.
        Adjusts bandwidth by the canonical kernel constant ratio.
    method : str
        ``'silverman'`` or ``'scott'``.

    Returns
    -------
    dict
        Keys: ``bandwidth``, ``sigma``, ``iqr``, ``method``, ``n_obs``.

    References
    ----------
    Silverman, B. W. (1986). Density Estimation for Statistics and Data
        Analysis. Chapman & Hall.
    Scott, D. W. (1992). Multivariate Density Estimation. Wiley.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    valid_methods = {"silverman", "scott"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {valid_methods}.")

    sigma = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))

    if method == "silverman":
        spread = min(sigma, iqr / 1.34) if iqr > 0 else sigma
        h = 1.06 * spread * n ** (-1 / 5)
    else:
        h = 1.059 * sigma * n ** (-1 / 5)

    kernel_ratios = {
        "gaussian": 1.0,
        "epanechnikov": 2.214,
        "uniform": 1.843,
    }
    if kernel in kernel_ratios:
        h *= kernel_ratios[kernel] / kernel_ratios["gaussian"]

    return {
        "bandwidth": h,
        "sigma": sigma,
        "iqr": iqr,
        "method": method,
        "n_obs": n,
    }


bwrot_fn = bwrot


def cheatsheet() -> str:
    return "bwrot({x}) -> Bandwidth selection via Silverman/Scott rule-of-thumb."
