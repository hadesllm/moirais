"""Block kriging for areal prediction (point-to-block)."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["spatial_block_kriging"]


def _cov_exp(h, c0, c1, a):
    h = np.asarray(h, dtype=float)
    return c1 * np.exp(-h / a) + np.where(h == 0, c0, 0.0)


def spatial_block_kriging(x, coords, blocks,
                          n_quad: int = 25,
                          nugget: float = 0.0, sill: float = 1.0,
                          range_: float = 1.0):
    """
    Ordinary block kriging: predict the average of Z over each block B
    via Monte Carlo / regular quadrature point integration of the
    point-to-block covariances.

    Predictor:  Z_hat(B) = lambda' Z,
    System:     [ C   1 ] [lambda]   [ C_bar(s_i, B) ]
                [ 1'  0 ] [  mu  ] = [       1       ]
    Variance:   sigma^2(B) = C_bar(B,B) - lambda' C_bar(., B) - mu.

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d)
    blocks : list of array-like
        Each element is a (q, d) array of quadrature points or a
        (lo, hi) tuple specifying a d-dimensional box; in the latter
        case `n_quad` regular quadrature nodes are generated per axis
        for d <= 2.
    n_quad : int
        Number of quadrature points per block when boxes are supplied.

    Returns
    -------
    RichResult with payload:
        estimate : list of block predictions
        se       : list of kriging SEs
        n, method
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = x.size
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match x ({n})")
    c0 = float(nugget); c1 = float(sill - nugget); a = float(range_)
    d = coords.shape[1]
    Dnn = cdist(coords, coords)
    C = _cov_exp(Dnn, c0, c1, a)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = C; A[:n, n] = 1.0; A[n, :n] = 1.0

    def block_to_pts(b):
        b_arr = np.asarray(b, dtype=float)
        if b_arr.ndim == 2 and b_arr.shape[0] == 2 and b_arr.shape[1] == d:
            lo = b_arr[0]; hi = b_arr[1]
            if d == 1:
                pts = np.linspace(lo[0], hi[0], n_quad).reshape(-1, 1)
                return pts
            if d == 2:
                k = int(round(np.sqrt(n_quad)))
                x1 = np.linspace(lo[0], hi[0], k)
                x2 = np.linspace(lo[1], hi[1], k)
                xx, yy = np.meshgrid(x1, x2)
                return np.column_stack([xx.ravel(), yy.ravel()])
            raise ValueError("box-form blocks supported only for d<=2")
        return b_arr  # explicit quad points

    ests = []; ses = []
    for b in blocks:
        pts = block_to_pts(b)
        # C_bar(s_i, B) -- average covariance from each datum to block points
        Cbar_iB = _cov_exp(cdist(coords, pts), c0, c1, a).mean(axis=1)
        Cbar_BB = float(_cov_exp(cdist(pts, pts), c0, c1, a).mean())
        rhs = np.append(Cbar_iB, 1.0)
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        lam = sol[:n]; mu = float(sol[n])
        z_hat = float(lam @ x)
        var = float(Cbar_BB - lam @ Cbar_iB - mu)
        ests.append(z_hat)
        ses.append(np.sqrt(max(var, 0.0)))

    return RichResult(payload={
        "estimate": ests,
        "se": ses,
        "n": int(n),
        "method": "Ordinary block kriging (exp. cov, MC quadrature)",
    })


def cheatsheet():
    return "spblk: Block kriging for areal prediction"


# CANONICAL TEST
# x=[1,2,3,4,5], coords=[[0],[1],[2],[3],[4]],
# blocks = [ [[0],[1]], [[3],[4]] ]  (two unit-length intervals)
# Expect estimate[0] ~ 1.5 (avg of z over [0,1]), estimate[1] ~ 3.5
