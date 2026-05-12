"""Leave-one-out cross-validation for ordinary kriging."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["spatial_cross_validation"]


def _cov_exp(h, c0, c1, a):
    h = np.asarray(h, dtype=float)
    return c1 * np.exp(-h / a) + np.where(h == 0, c0, 0.0)


def spatial_cross_validation(x, coords,
                              nugget: float = 0.0, sill: float = 1.0,
                              range_: float = 1.0):
    """
    Leave-one-out ordinary-kriging cross-validation.

    For each point i:
      - drop (x_i, s_i),
      - krige Z_hat_{-i}(s_i) using the remaining n-1,
      - record residual e_i = x_i - Z_hat_{-i}(s_i).

    Reported metrics:
      MSPE  = (1/n) sum e_i^2          (mean squared prediction error)
      RMSPE = sqrt(MSPE)
      MAE   = (1/n) sum |e_i|

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d)
    nugget, sill, range_ : float
        Exponential-covariance parameters used by each LOO predictor.

    Returns
    -------
    RichResult with payload: estimate (dict of MSPE/RMSPE/MAE/residuals),
        n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = x.size
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match x ({n})")
    c0 = float(nugget); c1 = float(sill - nugget); a = float(range_)
    D = cdist(coords, coords)
    resid = np.zeros(n)
    for i in range(n):
        sel = np.array([k for k in range(n) if k != i])
        Cii = _cov_exp(D[np.ix_(sel, sel)], c0, c1, a)
        c_vec = _cov_exp(D[i, sel], c0, c1, a)
        A = np.zeros((n, n))  # (n-1+1) augmented system
        m = n - 1
        A = np.zeros((m + 1, m + 1))
        A[:m, :m] = Cii
        A[:m, m] = 1.0; A[m, :m] = 1.0
        rhs = np.append(c_vec, 1.0)
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        z_hat = float(sol[:m] @ x[sel])
        resid[i] = x[i] - z_hat
    mspe = float((resid ** 2).mean())
    rmspe = float(np.sqrt(mspe))
    mae = float(np.abs(resid).mean())
    return RichResult(payload={
        "estimate": {
            "MSPE": mspe,
            "RMSPE": rmspe,
            "MAE": mae,
            "residuals": resid.tolist(),
        },
        "n": int(n),
        "method": "LOO cross-validation for ordinary kriging",
    })


def cheatsheet():
    return "spcrs: LOO cross-validation for ordinary kriging"


# CANONICAL TEST
# x = [1,2,3,4,5], coords = [[0],[1],[2],[3],[4]]
# nugget=0, sill=1, range_=2. MSPE > 0 (since LOO kriging is biased).
