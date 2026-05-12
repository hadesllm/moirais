"""Universal kriging with polynomial trend (Schabenberger & Gotway Ch 4)."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["universal_kriging"]


def _cov(h, c0, c1, a, model):
    h = np.asarray(h, dtype=float)
    if model == "exponential":
        return c1 * np.exp(-h / a) + np.where(h == 0, c0, 0.0)
    if model == "gaussian":
        return c1 * np.exp(-(h ** 2) / (a ** 2)) + np.where(h == 0, c0, 0.0)
    if model == "spherical":
        c = np.where(h <= a, c1 * (1.0 - 1.5 * h / a + 0.5 * (h / a) ** 3), 0.0)
        return c + np.where(h == 0, c0, 0.0)
    raise ValueError(f"unknown model: {model}")


def _trend(c, order):
    c = np.atleast_2d(c)
    if order == 0:
        return np.ones((c.shape[0], 1))
    if order == 1:
        return np.hstack([np.ones((c.shape[0], 1)), c])
    if order == 2:
        ones = np.ones((c.shape[0], 1))
        sq = c ** 2
        if c.shape[1] >= 2:
            cross = (c[:, 0:1] * c[:, 1:2])
            return np.hstack([ones, c, sq, cross])
        return np.hstack([ones, c, sq])
    raise ValueError("trend_order must be 0, 1, or 2")


def universal_kriging(x, coords, target,
                      model: str = "exponential",
                      nugget: float = 0.0, sill: float = 1.0, range_: float = 1.0,
                      trend_order: int = 1):
    """
    Universal kriging.

    Z(s) = mu(s) + delta(s),  mu(s) = sum_k beta_k f_k(s),
    delta(s) zero-mean second-order stationary with covariance C(h).
    Predictor solves the augmented system enforcing unbiasedness on
    every trend basis function.

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d)
    target : array-like, shape (m, d) or (d,)
    model : str
        Covariance model ('exponential' | 'gaussian' | 'spherical').
    nugget, sill, range_ : float
        Variogram parameters (c0, c0+c1, a).
    trend_order : int
        Polynomial trend order (0, 1, 2).

    Returns
    -------
    RichResult with payload:  estimate, se, n, method
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if target.shape[1] != coords.shape[1]:
        raise ValueError(
            f"target dim {target.shape[1]} must match coords dim {coords.shape[1]}")
    n = x.size
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match x ({n})")
    c0 = float(nugget); c1 = float(sill - nugget); a = float(range_)
    if c1 < 0:
        raise ValueError("sill must be >= nugget")

    Dnn = cdist(coords, coords)
    C = _cov(Dnn, c0, c1, a, model)
    F = _trend(coords, trend_order)
    p = F.shape[1]
    K = np.zeros((n + p, n + p))
    K[:n, :n] = C
    K[:n, n:] = F
    K[n:, :n] = F.T
    total_var = c0 + c1

    estimates = []
    ses = []
    for s0 in target:
        d0 = cdist(s0.reshape(1, -1), coords).ravel()
        c_vec = _cov(d0, c0, c1, a, model)
        f0 = _trend(s0.reshape(1, -1), trend_order).ravel()
        rhs = np.concatenate([c_vec, f0])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        lam = sol[:n]
        z_hat = float(lam @ x)
        var_pred = float(total_var - sol @ rhs)
        estimates.append(z_hat)
        ses.append(np.sqrt(max(var_pred, 0.0)))

    if target.shape[0] == 1:
        est_out, se_out = estimates[0], ses[0]
    else:
        est_out, se_out = estimates, ses
    return RichResult(payload={
        "estimate": est_out,
        "se": se_out,
        "n": int(n),
        "method": f"Universal kriging ({model}, trend_order={trend_order})",
    })


def cheatsheet():
    return "ukrig: Universal kriging with trend"


# CANONICAL TEST
# x=[1,2,3,4,5], coords=[[0],[1],[2],[3],[4]], target=[[2.5]], trend_order=1
# With a pure linear trend the predictor should recover the linear
# interpolation 3.5 (exactly) when nugget=0 since the trend basis
# already spans Z.
