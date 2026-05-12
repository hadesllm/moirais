"""Indicator kriging for exceedance probability."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["indicator_kriging"]


def _cov_exp(h, c0, c1, a):
    h = np.asarray(h, dtype=float)
    return c1 * np.exp(-h / a) + np.where(h == 0, c0, 0.0)


def indicator_kriging(x, coords, threshold,
                      target=None,
                      nugget: float = 0.0, sill: float = 0.25,
                      range_: float = 1.0):
    """
    Indicator kriging (Journel 1983; Schabenberger & Gotway Ch 4).

    For each point i, define I_i = 1(x_i <= threshold). Then ordinary-
    krige the indicator field at each target s0 to obtain
    P_hat(Z(s0) <= threshold).

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d)
    threshold : float
    target : array-like, optional
        Target coordinates, shape (m, d). Defaults to the sample
        coords (LOO-style probability map).
    nugget, sill, range_ : float
        Indicator-covariance parameters (sill near 0.25 is natural).

    Returns
    -------
    RichResult with payload: estimate (list of probabilities clipped to
        [0,1]), threshold, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = x.size
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match x ({n})")
    if target is None:
        target = coords
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if target.shape[1] != coords.shape[1]:
        raise ValueError("target/coords dim mismatch")

    I_obs = (x <= float(threshold)).astype(float)
    c0 = float(nugget); c1 = float(sill - nugget); a = float(range_)
    D = cdist(coords, coords)
    C = _cov_exp(D, c0, c1, a)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = C; A[:n, n] = 1.0; A[n, :n] = 1.0

    probs = []
    for s0 in target:
        d0 = cdist(s0.reshape(1, -1), coords).ravel()
        c_vec = _cov_exp(d0, c0, c1, a)
        rhs = np.append(c_vec, 1.0)
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        lam = sol[:n]
        p = float(np.clip(lam @ I_obs, 0.0, 1.0))
        probs.append(p)

    if len(probs) == 1:
        out = probs[0]
    else:
        out = probs
    return RichResult(payload={
        "estimate": out,
        "threshold": float(threshold),
        "n": int(n),
        "method": "Indicator kriging (ordinary, exp. cov)",
    })


def cheatsheet():
    return "indkr: Indicator kriging for probability mapping"


# CANONICAL TEST
# x = [1,2,3,4,5], coords = [[0],[1],[2],[3],[4]], threshold=3,
# target = [[2.5]], nugget=0, sill=0.25, range_=2.
# P(Z<=3) should be close to (#x<=3)/n = 3/5 = 0.6 but kriging may
# tilt it slightly.
