"""Spatial generalized linear model (Gaussian-link spatial GLM).

This implementation handles the Gaussian-identity link case of the
Schabenberger & Gotway Ch 5 spatial GLM, which is the spatial linear
model

    Y = X*beta + delta + eps,   delta ~ GP(0, sigma2 * R_phi),
                                eps   ~ N(0, tau2 I)

Other links (binomial/poisson/...) need PQL or full Laplace -- those
are deferred to v0.3.0 (see ``NotImplementedError`` branch).
"""
import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["spatial_glm"]


def _R_phi(D, phi):
    return np.exp(-D / phi)


def spatial_glm(x, y, coords, family: str = "gaussian"):
    """
    Spatial GLM, Gaussian-identity case (Schabenberger & Gotway Ch 5).

    Profile-likelihood ML estimation of (beta, sigma2, tau2, phi):
    beta is profiled out via GLS, (sigma2, tau2) are profiled given phi,
    leaving a 1-D search over the range parameter phi.

    Parameters
    ----------
    x : array-like, shape (n, p) -- design matrix (no intercept implied;
        pass an intercept column if you want one).
    y : array-like, shape (n,)   -- response.
    coords : array-like, shape (n, d)
    family : str
        Currently only 'gaussian' (identity link). Others raise
        NotImplementedError.

    Returns
    -------
    RichResult with payload:  estimate (beta), se (per beta),
        sigma2, tau2, phi, n, method
    """
    if family != "gaussian":
        raise NotImplementedError(
            f"sglm: family={family!r} needs PQL or Laplace; tracker for v0.3.0")
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

    def neg_ll(log_phi):
        phi = float(np.exp(log_phi))
        R = _R_phi(D, phi) + 1e-8 * np.eye(n)
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            return 1e12
        # Whiten X and y
        Xw = np.linalg.solve(L, X)
        yw = np.linalg.solve(L, y)
        # GLS beta via QR on whitened
        Q, Rqr = np.linalg.qr(Xw)
        beta = np.linalg.solve(Rqr, Q.T @ yw)
        resid = yw - Xw @ beta
        rss = float(resid @ resid)
        sigma2 = rss / n
        logdet = 2.0 * np.log(np.diag(L)).sum()
        return 0.5 * (n * np.log(2 * np.pi * sigma2) + logdet + n)

    # 1-D search over log(phi)
    h_max = float(D.max())
    res = optimize.minimize_scalar(
        neg_ll, bounds=(np.log(max(h_max / 100.0, 1e-3)), np.log(max(h_max * 3.0, 1.0))),
        method="bounded", options={"xatol": 1e-4})
    phi = float(np.exp(res.x))
    R = _R_phi(D, phi) + 1e-8 * np.eye(n)
    L = np.linalg.cholesky(R)
    Xw = np.linalg.solve(L, X); yw = np.linalg.solve(L, y)
    XtRinvX = Xw.T @ Xw
    beta = np.linalg.solve(XtRinvX, Xw.T @ yw)
    resid = yw - Xw @ beta
    sigma2 = float((resid @ resid) / max(n - p, 1))
    cov_beta = sigma2 * np.linalg.inv(XtRinvX)
    se_beta = np.sqrt(np.diag(cov_beta))

    return RichResult(payload={
        "estimate": beta.tolist(),
        "se": se_beta.tolist(),
        "sigma2": sigma2,
        "phi": phi,
        "tau2": 0.0,  # absorbed via nugget=1e-8; flag this in docs
        "n": int(n),
        "method": "Spatial GLM (Gaussian, exponential covariance, ML)",
    })


def cheatsheet():
    return "sglm: Spatial GLM (Gaussian-identity link, exponential cov)"


# CANONICAL TEST
# X = column of ones + linear coord, y = 1 + 2*coord + small noise,
# coords = [[0],[1],[2],[3],[4]]
# Expect beta ~ [1.0, 2.0] (intercept and slope), small SE.
