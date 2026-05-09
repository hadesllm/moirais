"""Eigenvector spatial filtering."""

import numpy as np

from ._containers import DescriptiveResult


def spatial_filter(y: np.ndarray, X: np.ndarray, W: np.ndarray, n_eigenvectors: int = 10) -> DescriptiveResult:
    """
    Eigenvector spatial filtering (ESF).

    Projects out spatial dependence by including eigenvectors of
    (I - 11'/n) W (I - 11'/n) as regressors.

    :param y: (n,) dependent variable.
    :param X: (n, k) explanatory variables.
    :param W: (n, n) spatial weights matrix.
    :param n_eigenvectors: Number of eigenvectors to include.
    :return: DescriptiveResult with coefficients and R-squared.

    References
    ----------
    Griffith DA (2003). Spatial Autocorrelation and Spatial Filtering.
    Springer.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    M = np.eye(n) - np.ones((n, n)) / n
    MWM = M @ W @ M
    eigvals, eigvecs = np.linalg.eigh(MWM)
    idx = np.argsort(np.abs(eigvals))[::-1]
    k = min(n_eigenvectors, n - 1)
    E = eigvecs[:, idx[:k]]
    X_aug = np.column_stack([X, E])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    fitted = X_aug @ beta
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="spatial_filter",
        value=r2,
        extra={
            "coefficients": beta[: X.shape[1]],
            "eigenvector_coeffs": beta[X.shape[1] :],
            "r_squared": r2,
            "n_eigenvectors": k,
            "n": n,
        },
    )


spflt = spatial_filter


def cheatsheet() -> str:
    return "spatial_filter({}) -> Eigenvector spatial filtering."
