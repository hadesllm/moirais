"""Vector autoregression (VAR) model fitting."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ["varmd", "var_model"]


def var_model(Y, p: int = 1) -> DescriptiveResult:
    """Fit a VAR(p) model via OLS (equation by equation).

    Model: ``Y[t] = c + A1 @ Y[t-1] + ... + Ap @ Y[t-p] + u[t]``

    Each equation is estimated by OLS independently, which is equivalent to
    GLS when the error covariance is unrestricted.

    Parameters
    ----------
    Y : array-like, shape (n,) or (n, m)
        Multivariate time series.  A 1-D input is treated as univariate (m=1).
    p : int
        Lag order >= 1.  Default 1.

    Returns
    -------
    DescriptiveResult
        value: float(AIC).
        extra keys:
          'coef'    : ndarray (m, 1 + m*p) — intercepts + lag coefficients
                      for each equation (row = equation).
          'sigma_u' : ndarray (m, m) — residual covariance matrix.
          'residuals': ndarray (n-p, m).
          'aic'     : float — Akaike information criterion.
          'bic'     : float — Bayesian information criterion.
          'n', 'p', 'm': dimensions.

    Raises
    ------
    ValueError
        If Y is not at least 2-D-compatible or too short for the lag order.

    References
    ----------
    Lütkepohl H. (2005). New Introduction to Multiple Time Series Analysis.
    Springer. Chapter 2.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 1-D or 2-D, got shape {Y.shape}.")
    n, m = Y.shape
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}.")
    n_params_per_eq = 1 + m * p
    if n - p <= n_params_per_eq:
        raise ValueError(
            f"Need n - p > 1 + m*p = {n_params_per_eq}; "
            f"got n={n}, p={p}, n-p={n - p}."
        )

    # Build regressor matrix Z: (n-p) x (1 + m*p).
    # Z[t] = [1, Y[t-1]', ..., Y[t-p]']
    n_use = n - p
    Z = np.empty((n_use, 1 + m * p), dtype=float)
    Z[:, 0] = 1.0
    for j in range(1, p + 1):
        Z[:, 1 + (j - 1) * m : 1 + j * m] = Y[p - j : n - j]

    Y_dep = Y[p:]  # (n_use, m)

    # OLS equation by equation: B = (Z'Z)^{-1} Z'Y_dep
    try:
        B, _, _, _ = np.linalg.lstsq(Z, Y_dep, rcond=None)
    except np.linalg.LinAlgError:
        B = np.zeros((1 + m * p, m))

    # Residuals and covariance (unbiased: divide by n_use - 1 - m*p).
    resid = Y_dep - Z @ B  # (n_use, m)
    df_res = n_use - (1 + m * p)
    if df_res > 0:
        sigma_u = (resid.T @ resid) / df_res
    else:
        sigma_u = np.eye(m) * 1e-8

    # Information criteria (Lütkepohl 2005, Ch. 4).
    k = 1 + m * p  # parameters per equation
    sign, logdet = np.linalg.slogdet(sigma_u)
    if sign <= 0:
        logdet = -np.inf
    aic = float(logdet + 2.0 * k * m / n_use)
    bic = float(logdet + np.log(float(n_use)) * k * m / n_use)

    return DescriptiveResult(
        name="var_model",
        value=float(aic),
        extra={
            "coef": B.T.copy(),       # (m, 1+m*p): row i = equation i
            "sigma_u": sigma_u.copy(),
            "residuals": resid.copy(),
            "aic": aic,
            "bic": bic,
            "n": n,
            "p": p,
            "m": m,
        },
    )


varmd = var_model


def cheatsheet() -> str:
    return "var_model(Y, p=1) -> VAR(p) model fit (OLS, equation-by-equation)."
