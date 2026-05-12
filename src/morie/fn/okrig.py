"""Ordinary kriging prediction (exact predictor with unbiasedness)."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["ordinary_kriging"]


def _covariance(h, c0, c1, a, model):
    """Stationary covariance C(h) = (c0 + c1) - gamma(h)."""
    h = np.asarray(h, dtype=float)
    if model == "exponential":
        return c1 * np.exp(-h / a) + np.where(h == 0, c0, 0.0)
    if model == "gaussian":
        return c1 * np.exp(-(h ** 2) / (a ** 2)) + np.where(h == 0, c0, 0.0)
    if model == "spherical":
        c = np.where(h <= a, c1 * (1.0 - 1.5 * h / a + 0.5 * (h / a) ** 3), 0.0)
        return c + np.where(h == 0, c0, 0.0)
    raise ValueError(f"unknown model: {model}")


def ordinary_kriging(x, coords, target,
                     model: str = "exponential",
                     nugget: float = 0.0, sill: float = 1.0, range_: float = 1.0):
    """
    Ordinary kriging prediction at one or more target locations.

    Solves the OK system:
        [ C   1 ] [ lambda ]   [ c0 ]
        [ 1'  0 ] [   mu   ] = [  1 ]
    with prediction  Z_hat(s0) = lambda' Z  and kriging variance
        sigma^2(s0) = (c0 + c1) - lambda' c0 - mu.

    Parameters
    ----------
    x : array-like, shape (n,)
        Observed values at sample locations.
    coords : array-like, shape (n, d)
        Sample coordinates.
    target : array-like, shape (m, d) or (d,)
        Prediction location(s).
    model : str
        Covariance model: 'exponential' | 'gaussian' | 'spherical'.
    nugget, sill, range_ : float
        Variogram parameters (c0, c0+c1, a). c1 = sill - nugget.

    Returns
    -------
    RichResult with payload:
        estimate : float or list of floats (kriged predictions)
        se       : float or list of floats (kriging standard errors)
        n, method
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
    c0 = float(nugget)
    c1 = float(sill - nugget)
    if c1 < 0:
        raise ValueError("sill must be >= nugget")
    a = float(range_)

    Dnn = cdist(coords, coords)
    C = _covariance(Dnn, c0, c1, a, model)  # (n,n)
    # Build augmented system A = [[C, 1], [1', 0]]
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = C
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    estimates = []
    ses = []
    total_var = c0 + c1  # C(0)
    for s0 in target:
        d0 = cdist(s0.reshape(1, -1), coords).ravel()
        c_vec = _covariance(d0, c0, c1, a, model)
        rhs = np.append(c_vec, 1.0)
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        lam = sol[:n]
        mu = sol[n]
        z_hat = float(lam @ x)
        var_pred = float(total_var - lam @ c_vec - mu)
        var_pred = max(var_pred, 0.0)
        estimates.append(z_hat)
        ses.append(np.sqrt(var_pred))

    if target.shape[0] == 1:
        est_out, se_out = estimates[0], ses[0]
    else:
        est_out, se_out = estimates, ses
    return RichResult(payload={
        "estimate": est_out,
        "se": se_out,
        "n": int(n),
        "method": f"Ordinary kriging ({model})",
    })


def cheatsheet():
    return "okrig: Ordinary kriging prediction"


# CANONICAL TEST
# x=[1,2,3,4,5], coords=[[0],[1],[2],[3],[4]], target=[[2.5]]
# model='exponential', nugget=0, sill=1, range_=2
# At s0=2.5 the OK predictor should give ~3.5 (between z2=3 and z3=4)
# and a non-negative kriging variance.
