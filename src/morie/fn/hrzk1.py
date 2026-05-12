# morie.fn — function file (hadesllm/morie)
"""Kernel density estimator for semiparametric models (Horowitz 2009, Ch 2).

Implements the standard univariate Rosenblatt-Parzen kernel density estimator

    f_hat(x) = (1 / (n h)) * sum_i K((x - X_i)/h)

with a Gaussian kernel.  If ``x`` is a scalar the estimator and its
asymptotic standard error are returned at that point; if ``x`` is an
array the estimator is returned on a sample grid (or at each point in
``x`` when ``grid=None``).

Asymptotic SE at a point ``x0``:
    var(f_hat(x0)) ≈ f(x0) * R(K) / (n h),
    R(K_gaussian) = 1/(2*sqrt(pi)) ≈ 0.2820948.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_kernel_density"]

_R_K_GAUSSIAN = 1.0 / (2.0 * np.sqrt(np.pi))  # roughness of the Gaussian kernel


def _silverman_bandwidth(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = max(s, 1e-6)
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def horowitz_kernel_density(x, bandwidth=None, sample=None):
    """Rosenblatt-Parzen kernel density estimator.

    Parameters
    ----------
    x : float or array-like
        Either a scalar point at which to evaluate f_hat (when ``sample``
        is provided), OR the data sample itself (when ``sample is None``).
    bandwidth : float, optional
        Kernel bandwidth h.  Defaults to Silverman's rule of thumb.
    sample : array-like, optional
        Data sample used to estimate the density.  When omitted, ``x`` is
        treated as both the data and the evaluation grid.

    Returns
    -------
    RichResult with payload keys: estimate, se, bandwidth, n, kernel, method
    """
    if sample is None:
        data = np.asarray(x, dtype=float).ravel()
        grid = data.copy()
    else:
        data = np.asarray(sample, dtype=float).ravel()
        grid = np.atleast_1d(np.asarray(x, dtype=float))
    n = data.size
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "kernel-density (insufficient data)"})
    h = float(bandwidth) if bandwidth is not None else _silverman_bandwidth(data)
    if h <= 0:
        h = _silverman_bandwidth(data)
    # f_hat on grid: each row = one eval point, col = data point
    diffs = (grid[:, None] - data[None, :]) / h
    weights = np.exp(-0.5 * diffs * diffs) / np.sqrt(2 * np.pi)
    f_hat = weights.mean(axis=1) / h
    # Asymptotic pointwise SE
    se = np.sqrt(np.maximum(f_hat, 0) * _R_K_GAUSSIAN / (n * h))
    if f_hat.size == 1:
        est = float(f_hat[0]); se_v = float(se[0])
    else:
        est = f_hat.astype(float); se_v = se.astype(float)
    return RichResult(payload={
        "estimate": est, "se": se_v, "bandwidth": h, "n": n,
        "kernel": "gaussian",
        "method": "Rosenblatt-Parzen kernel density (Horowitz 2009, Ch 2)",
    })


def cheatsheet():
    return "hrzk1: kernel density estimator (Gaussian, Silverman bw)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 500)
    res = horowitz_kernel_density(0.0, sample=s)
    print(res)
    assert abs(res["estimate"] - 0.3989) < 0.1
