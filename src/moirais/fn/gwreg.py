# moirais.fn — function file (hadesllm/moirais)
"""Geographically weighted regression (GWR)."""

import numpy as np

from ._containers import DescriptiveResult


def gwr(
    y: np.ndarray, X: np.ndarray, coordinates: np.ndarray, bandwidth: float | None = None, kernel: str = "gaussian"
) -> DescriptiveResult:
    """
    Geographically weighted regression (GWR).

    Fits a local linear regression at each observation using
    distance-weighted neighbours.

    :param y: (n,) dependent variable.
    :param X: (n, k) explanatory variables.
    :param coordinates: (n, 2) spatial coordinates.
    :param bandwidth: Kernel bandwidth (default: median distance).
    :param kernel: Kernel type: 'gaussian' or 'bisquare'.
    :return: DescriptiveResult with local coefficients and R-squared.

    References
    ----------
    Brunsdon C, Fotheringham AS, Charlton ME (1996). Geographically
    weighted regression: a method for exploring spatial nonstationarity.
    Geographical Analysis, 28(4), 281-298.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    k = X.shape[1]
    dmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    if bandwidth is None:
        bandwidth = float(np.median(dmat[dmat > 0]))
    local_betas = np.zeros((n, k))
    local_r2 = np.zeros(n)
    for i in range(n):
        d = dmat[i]
        if kernel == "bisquare":
            w = np.where(d <= bandwidth, (1 - (d / bandwidth) ** 2) ** 2, 0.0)
        else:
            w = np.exp(-0.5 * (d / bandwidth) ** 2)
        W = np.diag(w)
        XtWX = X.T @ W @ X
        try:
            local_betas[i] = np.linalg.solve(XtWX, X.T @ W @ y)
        except np.linalg.LinAlgError:
            local_betas[i] = np.linalg.lstsq(XtWX, X.T @ W @ y, rcond=None)[0]
        fitted_i = X[i] @ local_betas[i]
        ss_res = w @ (y - X @ local_betas[i]) ** 2
        ss_tot = w @ (y - np.average(y, weights=w)) ** 2
        local_r2[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="gwr",
        value=float(local_r2.mean()),
        extra={
            "local_betas": local_betas,
            "local_r2": local_r2,
            "bandwidth": bandwidth,
            "kernel": kernel,
            "n": n,
            "k": k,
        },
    )


gwreg = gwr


def cheatsheet() -> str:
    return "gwr({}) -> Geographically weighted regression (GWR)."
