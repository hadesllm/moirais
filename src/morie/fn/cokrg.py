"""Cokriging (linear model of coregionalization, simple-cokriging form)."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["cokriging"]


def _cov_exp(h, c0, c1, a):
    h = np.asarray(h, dtype=float)
    return c1 * np.exp(-h / a) + np.where(h == 0, c0, 0.0)


def cokriging(x, y, coords, target,
              sill_p: float = 1.0, range_p: float = 1.0,
              sill_s: float = 1.0, range_s: float = 1.0,
              cross_sill: float = 0.5, cross_range: float = 1.0,
              nugget: float = 0.0):
    """
    Simple cokriging on co-located primary/secondary observations.

    System:
       [ Cpp  Cps ] [ lambda ]   [ c_0p ]
       [ Cps' Css ] [  mu    ] = [ c_0s ]
    Predictor:  Z1_hat(s0) = lambda' Z1 + mu' Z2.

    Parameters
    ----------
    x : array-like, shape (n,)   -- primary variable Z1.
    y : array-like, shape (n,)   -- secondary variable Z2.
    coords : array-like, shape (n, d) -- co-located sample coords.
    target : array-like, shape (m, d) or (d,)
    sill_p, range_p : float -- primary auto-covariance parameters.
    sill_s, range_s : float -- secondary auto-covariance parameters.
    cross_sill, cross_range : float -- cross-covariance parameters.
    nugget : float

    Returns
    -------
    RichResult with payload: estimate, se, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    n = x.size
    if y.size != n or coords.shape[0] != n:
        raise ValueError("x, y, and coords must have matching n")
    if target.shape[1] != coords.shape[1]:
        raise ValueError("target/coords dim mismatch")

    D = cdist(coords, coords)
    Cpp = _cov_exp(D, nugget, sill_p - nugget, range_p)
    Css = _cov_exp(D, nugget, sill_s - nugget, range_s)
    Cps = cross_sill * np.exp(-D / cross_range)
    C = np.block([[Cpp, Cps], [Cps.T, Css]])
    z = np.concatenate([x, y])
    var0 = sill_p  # C_11(0)

    ests = []; ses = []
    for s0 in target:
        d0 = cdist(s0.reshape(1, -1), coords).ravel()
        c0p = _cov_exp(d0, nugget, sill_p - nugget, range_p)
        c0s = cross_sill * np.exp(-d0 / cross_range)
        c0 = np.concatenate([c0p, c0s])
        try:
            w = np.linalg.solve(C, c0)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(C, c0, rcond=None)[0]
        z_hat = float(w @ z)
        var = float(var0 - w @ c0)
        ests.append(z_hat)
        ses.append(np.sqrt(max(var, 0.0)))

    if target.shape[0] == 1:
        ests_out, ses_out = ests[0], ses[0]
    else:
        ests_out, ses_out = ests, ses
    return RichResult(payload={
        "estimate": ests_out,
        "se": ses_out,
        "n": int(n),
        "method": "Simple cokriging (linear coregionalization, exp. cov)",
    })


def cheatsheet():
    return "cokrg: Cokriging for multivariate spatial prediction"


# CANONICAL TEST
# x = [1,2,3,4,5], y = [2,4,6,8,10],  coords = [[0],[1],[2],[3],[4]],
# target = [[2.5]];  expect estimate ~ 3.5
