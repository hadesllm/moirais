"""Thin-plate spline regression (Duchon 1977).

For data ``(x_i, y_i)`` in R^d, fits

    f(x) = sum_i a_i phi(||x - x_i||) + beta_0 + beta' x

with radial basis ``phi(r) = r^2 log r`` (2-D classical TPS).  Solves
the saddle-point system

    | K  T | |a|   |y|
    | T' 0 | |b| = |0|

where K_ij = phi(||x_i - x_j||) and T = [1 x].
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["thin_plate_spline"]


def _phi(r):
    out = np.zeros_like(r)
    mask = r > 0
    out[mask] = (r[mask] ** 2) * np.log(r[mask])
    return out


def thin_plate_spline(x, y, lam: float = 0.0):
    """Thin-plate spline regression.

    Parameters
    ----------
    x : (n,) or (n,d) array
    y : (n,) array
    lam : float
        Smoothing penalty; lam=0 = exact interpolation.

    Returns
    -------
    RichResult: a, beta, fitted, residuals, r2, lam, n, method.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    if n < d + 2 or y.size != n:
        return RichResult(payload={"estimate": float("nan"), "n": int(n),
                                   "method": "TPS (n too small)"})
    # pairwise distances
    diff = x[:, None, :] - x[None, :, :]
    R = np.sqrt(np.sum(diff ** 2, axis=2))
    K = _phi(R) + lam * np.eye(n)
    T = np.column_stack([np.ones(n), x])
    A = np.block([[K, T], [T.T, np.zeros((d + 1, d + 1))]])
    rhs = np.concatenate([y, np.zeros(d + 1)])
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a = sol[:n]
    beta = sol[n:]
    fitted = K @ a + T @ beta
    resid = y - fitted
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return RichResult(payload={
        "a": a, "beta": beta, "fitted": fitted, "residuals": resid,
        "sse": sse, "r2": float(r2), "lambda": float(lam),
        "estimate": float(fitted.mean()),
        "n": int(n), "d": int(d),
        "method": "Thin-plate spline (Duchon 1977)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.uniform(0, 1, (40, 2))
# >>> y = x[:,0] + x[:,1]
# >>> res = thin_plate_spline(x, y, lam=1e-8)
# >>> assert res["r2"] > 0.99  # linear surface is exactly representable


def cheatsheet():
    return "tpspn(x, y, lam=0): thin-plate spline regression."
