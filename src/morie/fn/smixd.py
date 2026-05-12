"""Spatial linear mixed model (REML).

Y = X*beta + delta + eps,
   delta ~ N(0, sigma2 * R_phi),  R_phi exponential,
   eps   ~ N(0, tau2 * I).

REML estimation profiles beta out and maximizes the restricted
log-likelihood over (phi, nu = tau2/sigma2); sigma2 is profiled in
closed form (Patterson-Thompson 1971).
"""
import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["spatial_mixed_model"]


def _Sigma(D, phi, nu):
    return np.exp(-D / phi) + nu * np.eye(D.shape[0])


def spatial_mixed_model(x, y, coords):
    """
    REML estimation of the spatial linear mixed model.

    Parameters
    ----------
    x : array-like, shape (n, p)
    y : array-like, shape (n,)
    coords : array-like, shape (n, d)

    Returns
    -------
    RichResult with payload: estimate (beta), se, sigma2, tau2, phi,
        n, method.
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n, p = X.shape
    if y.size != n or coords.shape[0] != n:
        raise ValueError("shape mismatch among x, y, coords")
    D = cdist(coords, coords)
    h_max = float(max(D.max(), 1.0))

    def neg_reml(theta):
        log_phi, log_nu = theta
        phi = np.exp(log_phi); nu = np.exp(log_nu)
        Sigma = _Sigma(D, phi, nu)
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            return 1e12
        Xw = np.linalg.solve(L, X); yw = np.linalg.solve(L, y)
        XtSiX = Xw.T @ Xw
        try:
            L2 = np.linalg.cholesky(XtSiX)
        except np.linalg.LinAlgError:
            return 1e12
        beta = np.linalg.solve(L2.T, np.linalg.solve(L2, Xw.T @ yw))
        resid = yw - Xw @ beta
        rss = float(resid @ resid)
        sigma2 = rss / max(n - p, 1)
        logdet_Sigma = 2.0 * np.log(np.diag(L)).sum()
        logdet_XtSiX = 2.0 * np.log(np.diag(L2)).sum()
        return 0.5 * (logdet_Sigma + logdet_XtSiX +
                      (n - p) * np.log(2 * np.pi * sigma2) + (n - p))

    x0 = np.array([np.log(h_max / 3.0), np.log(0.1)])
    res = optimize.minimize(
        neg_reml, x0, method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 400})
    phi = float(np.exp(res.x[0])); nu = float(np.exp(res.x[1]))
    Sigma = _Sigma(D, phi, nu)
    L = np.linalg.cholesky(Sigma)
    Xw = np.linalg.solve(L, X); yw = np.linalg.solve(L, y)
    XtSiX = Xw.T @ Xw
    beta = np.linalg.solve(XtSiX, Xw.T @ yw)
    resid = yw - Xw @ beta
    sigma2 = float((resid @ resid) / max(n - p, 1))
    tau2 = float(nu * sigma2)
    cov_beta = sigma2 * np.linalg.inv(XtSiX)
    se_beta = np.sqrt(np.diag(cov_beta))

    return RichResult(payload={
        "estimate": beta.tolist(),
        "se": se_beta.tolist(),
        "sigma2": sigma2,
        "tau2": tau2,
        "phi": phi,
        "n": int(n),
        "method": "Spatial linear mixed model (REML, exponential covariance)",
    })


def cheatsheet():
    return "smixd: Spatial linear mixed model (REML)"


# CANONICAL TEST
# X = [[1,0],[1,1],[1,2],[1,3],[1,4]],  y = [1,3,5,7,9],
# coords = [[0],[1],[2],[3],[4]]
# Expect beta ~ [1, 2] (intercept, slope).
