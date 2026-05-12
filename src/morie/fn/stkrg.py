"""Spatiotemporal ordinary kriging (separable covariance)."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["spatiotemporal_kriging"]


def spatiotemporal_kriging(x, coords, times, target,
                           sill: float = 1.0, nugget: float = 0.0,
                           range_s: float = 1.0, range_t: float = 1.0):
    """
    Ordinary spatiotemporal kriging with a separable exponential
    covariance:

        C((h, u)) = (sill - nugget) * exp(-h/range_s) * exp(-|u|/range_t)
                  + nugget if h == 0 and u == 0.

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d_s)
    times  : array-like, shape (n,)
    target : tuple (s0, t0), where s0 is (m, d_s) and t0 is (m,)
        (or scalars for a single target).
    sill, nugget, range_s, range_t : float

    Returns
    -------
    RichResult with payload: estimate, se, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    t = np.asarray(times, dtype=float).ravel()
    n = x.size
    if coords.shape[0] != n or t.size != n:
        raise ValueError("shape mismatch among x, coords, times")
    s0, t0 = target
    s0 = np.asarray(s0, dtype=float)
    if s0.ndim == 1:
        s0 = s0.reshape(1, -1)
    t0 = np.atleast_1d(np.asarray(t0, dtype=float))
    if s0.shape[1] != coords.shape[1]:
        raise ValueError("target coords dim mismatch")
    if s0.shape[0] != t0.size:
        raise ValueError("target s0 and t0 must align")
    c0 = float(nugget); c1 = float(sill - nugget)
    Dnn = cdist(coords, coords)
    Tnn = np.abs(t[:, None] - t[None, :])
    C = c1 * np.exp(-Dnn / range_s) * np.exp(-Tnn / range_t) + c0 * np.eye(n)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = C; A[:n, n] = 1.0; A[n, :n] = 1.0
    var_total = sill

    ests = []; ses = []
    for k in range(s0.shape[0]):
        d0s = cdist(s0[k:k+1], coords).ravel()
        d0t = np.abs(t0[k] - t)
        c_vec = c1 * np.exp(-d0s / range_s) * np.exp(-d0t / range_t)
        rhs = np.append(c_vec, 1.0)
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        lam = sol[:n]; mu = float(sol[n])
        z_hat = float(lam @ x)
        var = float(var_total - lam @ c_vec - mu)
        ests.append(z_hat)
        ses.append(np.sqrt(max(var, 0.0)))

    if s0.shape[0] == 1:
        est_out, se_out = ests[0], ses[0]
    else:
        est_out, se_out = ests, ses
    return RichResult(payload={
        "estimate": est_out,
        "se": se_out,
        "n": int(n),
        "method": "Spatiotemporal ordinary kriging (separable exponential)",
    })


def cheatsheet():
    return "stkrg: Spatiotemporal kriging"


# CANONICAL TEST
# x = [i + t for (i,t) in pairs], coords = [[0],[1],[2]], times = [0,1,2]
# (3 points) target = ([[1]], [1]) -> exact obs at (1,1) gives 2.0
