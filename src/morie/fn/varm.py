"""VAR (vector autoregression) model."""

import numpy as np

from ._containers import DescriptiveResult


def var_fit(Y: np.ndarray, lags: int = 1) -> DescriptiveResult:
    """
    Fit a VAR(p) model via OLS.

    .. math::

        \\mathbf{y}_t = \\mathbf{c} + A_1 \\mathbf{y}_{t-1}
        + \\cdots + A_p \\mathbf{y}_{t-p} + \\mathbf{u}_t

    :param Y: (n, k) array of k endogenous time series.
    :param lags: Number of lags p. Default 1.
    :return: DescriptiveResult with coefficient matrices, residuals,
        covariance.
    :raises ValueError: If insufficient observations.

    References
    ----------
    Sims C.A. (1980). Macroeconomics and reality. *Econometrica*,
    48(1), 1-48.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, k = Y.shape
    if n < lags + 3:
        raise ValueError(f"Need at least {lags + 3} observations, got {n}.")
    T = n - lags
    dep = Y[lags:]
    regs = [np.ones((T, 1))]
    for lag in range(1, lags + 1):
        regs.append(Y[lags - lag : n - lag])
    X = np.hstack(regs)
    B = np.linalg.lstsq(X, dep, rcond=None)[0]
    residuals = dep - X @ B
    sigma_u = residuals.T @ residuals / T
    intercept = B[0]
    coef_matrices = []
    for lag in range(lags):
        coef_matrices.append(B[1 + lag * k : 1 + (lag + 1) * k])
    return DescriptiveResult(
        name="var_fit",
        value=float(np.linalg.det(sigma_u)),
        extra={
            "intercept": intercept.tolist(),
            "coefficients": [c.tolist() for c in coef_matrices],
            "sigma_u": sigma_u.tolist(),
            "residuals": residuals,
            "lags": lags,
            "k": k,
            "n": n,
            "T": T,
        },
    )


varm = var_fit


def cheatsheet() -> str:
    return "var_fit({}) -> VAR(p) vector autoregression via OLS."
