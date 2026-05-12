# morie.fn -- function file (hadesllm/morie)
"""Isotonic (monotone) regression via the pool-adjacent-violators algorithm
(Barlow, Bartholomew, Bremner & Brunk 1972).

Solves

    min_f  sum_i w_i (y_i - f_i)^2     s.t. f_1 <= f_2 <= ... <= f_n

where ``f_i = f(x_i)`` and the x's are sorted.  Uses scikit-learn's
``IsotonicRegression`` when available, else a pure-numpy PAVA fallback.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["isotonic_regression"]


def _pava(y, w):
    """Pool-adjacent-violators (non-decreasing); pure numpy reference."""
    y = np.asarray(y, dtype=float).copy()
    w = np.asarray(w, dtype=float).copy()
    n = y.size
    # block representation: (start, end, mean, weight)
    means = y.copy()
    weights = w.copy()
    # active blocks via two pointers
    i = 0
    blocks = [[i, i, means[i], weights[i]] for i in range(n)]
    out_blocks = []
    for b in blocks:
        out_blocks.append(b)
        while (len(out_blocks) >= 2
               and out_blocks[-2][2] >= out_blocks[-1][2]):
            a = out_blocks.pop()
            prev = out_blocks.pop()
            new_w = prev[3] + a[3]
            new_m = (prev[2] * prev[3] + a[2] * a[3]) / new_w
            out_blocks.append([prev[0], a[1], new_m, new_w])
    fitted = np.empty(n)
    for s, e, m, _w in out_blocks:
        fitted[s:e + 1] = m
    return fitted


def isotonic_regression(x, y, weights=None, increasing: bool = True):
    """Isotonic regression (monotone least-squares fit).

    Parameters
    ----------
    x : (n,) array
        Predictor; sample is sorted internally.
    y : (n,) array
    weights : (n,) array, optional
    increasing : bool
        If False, fits a non-increasing fit (reverse, then PAVA).

    Returns
    -------
    RichResult: x_sorted, fitted, residuals, r2, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < 2 or y.size != n:
        return RichResult(payload={"estimate": float("nan"), "n": int(n),
                                   "method": "Isotonic (n<2)"})
    if weights is None:
        weights = np.ones(n)
    order = np.argsort(x)
    xs = x[order]; ys = y[order]; ws = np.asarray(weights, dtype=float)[order]
    try:
        from sklearn.isotonic import IsotonicRegression
        fit = IsotonicRegression(increasing=increasing).fit(xs, ys,
                                                            sample_weight=ws)
        fitted = fit.predict(xs)
    except Exception:
        fitted = _pava(ys, ws) if increasing else _pava(-ys, ws) * -1
    resid = ys - fitted
    sse = float(np.sum(ws * resid ** 2))
    sst = float(np.sum(ws * (ys - np.average(ys, weights=ws)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return RichResult(payload={
        "x_sorted": xs, "fitted": fitted, "residuals": resid,
        "sse": sse, "r2": float(r2), "estimate": float(fitted.mean()),
        "n": int(n),
        "method": "Isotonic regression (Barlow et al. 1972, PAVA)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> x = np.arange(10, dtype=float)
# >>> y = np.array([1, 3, 2, 5, 4, 6, 7, 8, 7, 10], dtype=float)
# >>> res = isotonic_regression(x, y)
# >>> # fitted values must be non-decreasing
# >>> assert np.all(np.diff(res["fitted"]) >= -1e-9)


def cheatsheet():
    return "isotn(x, y, weights=None, increasing=True): PAVA isotonic fit."
