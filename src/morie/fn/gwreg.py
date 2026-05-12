"""Geographically weighted regression (GWR)."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["geographically_weighted_regression"]


def geographically_weighted_regression(x, y, coords,
                                       bandwidth: float | None = None,
                                       kernel: str = "gaussian"):
    """
    Geographically weighted regression (Brunsdon, Fotheringham, Charlton 1996).

    At each sample location i, fit a local weighted least squares:
        beta(s_i) = (X' W(s_i) X)^{-1} X' W(s_i) y,
    where W(s_i) is a kernel-weighted diagonal of distances from i.

    Parameters
    ----------
    x : array-like, shape (n, k)
        Design matrix (include intercept column if desired).
    y : array-like, shape (n,)
    coords : array-like, shape (n, d)
    bandwidth : float, optional
        Kernel bandwidth (default: median pairwise distance).
    kernel : str
        'gaussian' (default) or 'bisquare'.

    Returns
    -------
    RichResult with payload: estimate (n x k local betas as nested list),
        se (n x k local SEs), bandwidth, kernel, n, method.
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n, k = X.shape
    if y.size != n or coords.shape[0] != n:
        raise ValueError("shape mismatch among x, y, coords")
    D = cdist(coords, coords)
    if bandwidth is None:
        bandwidth = float(np.median(D[D > 0]))
    if kernel not in ("gaussian", "bisquare"):
        raise ValueError(f"unknown kernel: {kernel}")

    local_betas = np.zeros((n, k))
    local_ses = np.zeros((n, k))
    for i in range(n):
        d = D[i]
        if kernel == "bisquare":
            w = np.where(d <= bandwidth, (1.0 - (d / bandwidth) ** 2) ** 2, 0.0)
        else:
            w = np.exp(-0.5 * (d / bandwidth) ** 2)
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]; yw = y * sqrt_w
        XtWX = Xw.T @ Xw
        try:
            beta_i = np.linalg.solve(XtWX, Xw.T @ yw)
        except np.linalg.LinAlgError:
            beta_i = np.linalg.lstsq(XtWX, Xw.T @ yw, rcond=None)[0]
        local_betas[i] = beta_i
        resid = yw - Xw @ beta_i
        df = max(w.sum() - k, 1.0)
        sigma2_i = float(resid @ resid) / df
        try:
            cov_i = sigma2_i * np.linalg.inv(XtWX)
            local_ses[i] = np.sqrt(np.maximum(np.diag(cov_i), 0.0))
        except np.linalg.LinAlgError:
            local_ses[i] = np.nan

    return RichResult(payload={
        "estimate": local_betas.tolist(),
        "se": local_ses.tolist(),
        "bandwidth": float(bandwidth),
        "kernel": kernel,
        "n": int(n),
        "method": f"GWR ({kernel} kernel)",
    })


def cheatsheet():
    return "gwreg: Geographically weighted regression"


# CANONICAL TEST
# X = [[1,0],[1,1],[1,2],[1,3],[1,4]],  y = [1,2,3,4,5],
# coords = [[0],[1],[2],[3],[4]]
# Linear truth y = 0 + 1*coord -> every local beta ~ [0, 1].
