# morie.fn — function file (hadesllm/morie)
"""Nadaraya-Watson kernel regression (Horowitz 2009, Ch 2).

    m_hat(x) = sum_i K_h(x - X_i) Y_i / sum_i K_h(x - X_i)

with a Gaussian kernel.  Returns a fitted value at each X_i (leave-one-in
fit) plus an asymptotic pointwise SE based on local variance
``sigma_hat^2 = sum w_i (y - m_hat)^2``.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_kernel_regression"]

_R_K_GAUSSIAN = 1.0 / (2.0 * np.sqrt(np.pi))


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


def horowitz_kernel_regression(x, y, bandwidth=None, grid=None):
    """Nadaraya-Watson kernel regression.

    Parameters
    ----------
    x, y : array-like
        Univariate regressor and response.
    bandwidth : float, optional
        Gaussian bandwidth (default Silverman ROT).
    grid : array-like, optional
        Evaluation points.  Default: the X sample itself.

    Returns
    -------
    RichResult with payload keys: estimate, se, bandwidth, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < 2 or y.size != n:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "nadaraya-watson (insufficient data)"})
    h = float(bandwidth) if bandwidth is not None else _silverman_bandwidth(x)
    if h <= 0:
        h = _silverman_bandwidth(x)
    g = x if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    u = (g[:, None] - x[None, :]) / h
    w = np.exp(-0.5 * u * u)            # un-normalised; ratio cancels constants
    wsum = w.sum(axis=1)
    safe_wsum = np.where(wsum > 0, wsum, 1.0)
    m_hat = (w @ y) / safe_wsum
    # Local conditional variance estimator
    resid = y[None, :] - m_hat[:, None]
    sigma2 = (w * resid * resid).sum(axis=1) / safe_wsum
    # Pointwise asymptotic SE  ~ sqrt(sigma2 * R(K) / (n h f(x)))
    # use f_hat(x) via the same weights / (n h * sqrt(2 pi))
    f_hat = wsum / (n * h * np.sqrt(2 * np.pi))
    se = np.sqrt(np.maximum(sigma2, 0) * _R_K_GAUSSIAN /
                 (n * h * np.maximum(f_hat, 1e-12)))
    if m_hat.size == 1:
        est = float(m_hat[0]); se_v = float(se[0])
    else:
        est = m_hat.astype(float); se_v = se.astype(float)
    return RichResult(payload={
        "estimate": est, "se": se_v, "bandwidth": h, "n": n,
        "method": "Nadaraya-Watson kernel regression (Gaussian)",
    })


def cheatsheet():
    return "hrzk2: Nadaraya-Watson kernel regression"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 300)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(300)
    res = horowitz_kernel_regression(x, y, grid=[0.25, 0.5, 0.75])
    print(res)
    assert abs(res["estimate"][0] - 1.0) < 0.3
    assert abs(res["estimate"][2] - (-1.0)) < 0.3
