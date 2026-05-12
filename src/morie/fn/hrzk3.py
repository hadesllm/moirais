# morie.fn — function file (hadesllm/morie)
"""Local-linear regression estimator (Horowitz 2009, Ch 2).

    (a_hat, b_hat) = argmin_{a,b} sum_i K_h(x - X_i) (Y_i - a - b (X_i - x))^2

Returns the local intercept ``a_hat(x)`` (the estimate of m(x)) plus an
asymptotic pointwise SE.  Uses a Gaussian kernel and Silverman bandwidth
when not supplied.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_local_linear"]

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


def horowitz_local_linear(x, y, bandwidth=None, grid=None):
    """Local-linear regression on a univariate ``x``.

    Returns the local-intercept fit at each ``grid`` point (or each sample
    point if ``grid`` is omitted) together with pointwise SEs.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < 3 or y.size != n:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "local-linear (insufficient data)"})
    h = float(bandwidth) if bandwidth is not None else _silverman_bandwidth(x)
    if h <= 0:
        h = _silverman_bandwidth(x)
    g = x if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    m_hat = np.zeros(g.size)
    se = np.zeros(g.size)
    for i, x0 in enumerate(g):
        u = (x - x0) / h
        w = np.exp(-0.5 * u * u)
        if w.sum() <= 1e-12:
            m_hat[i] = np.nan; se[i] = np.nan; continue
        X = np.column_stack([np.ones(n), x - x0])
        WX = X * w[:, None]
        XtWX = X.T @ WX
        try:
            beta = np.linalg.solve(XtWX, WX.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtWX) @ (WX.T @ y)
        m_hat[i] = beta[0]
        resid = y - X @ beta
        sigma2 = float((w * resid * resid).sum() / max(w.sum(), 1e-12))
        f_hat = w.sum() / (n * h * np.sqrt(2 * np.pi))
        se[i] = np.sqrt(max(sigma2, 0) * _R_K_GAUSSIAN /
                        (n * h * max(f_hat, 1e-12)))
    if m_hat.size == 1:
        est = float(m_hat[0]); se_v = float(se[0])
    else:
        est = m_hat.astype(float); se_v = se.astype(float)
    return RichResult(payload={
        "estimate": est, "se": se_v, "bandwidth": h, "n": n,
        "method": "Local-linear regression (Gaussian kernel)",
    })


def cheatsheet():
    return "hrzk3: local-linear regression"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 400)
    y = 2 * x + 0.05 * rng.standard_normal(400)
    res = horowitz_local_linear(x, y, grid=[0.0])
    print(res)
    assert abs(res["estimate"] - 0.0) < 0.15
