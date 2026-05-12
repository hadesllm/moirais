"""Vector Error Correction Model (VECM) for cointegrated series."""

import numpy as np


def vecm(
    y1: np.ndarray,
    y2: np.ndarray,
    lags: int = 1,
) -> dict:
    r"""
    Vector Error Correction Model for two cointegrated series.

    Estimates the VECM:

    .. math::

        \\Delta \\mathbf{y}_t = \\boldsymbol{\\alpha} \\boldsymbol{\\beta}'
        \\mathbf{y}_{t-1}
        + \\sum_{i=1}^{p-1} \\boldsymbol{\\Gamma}_i \\Delta \\mathbf{y}_{t-i}
        + \\boldsymbol{\\varepsilon}_t

    where :math:`\\boldsymbol{\\alpha}` gives adjustment speeds and
    :math:`\\boldsymbol{\\beta}` the cointegrating vector.

    :param y1: 1-D array for the first series.
    :param y2: 1-D array for the second series (same length as *y1*).
    :param lags: Number of lagged difference terms. Default 1.
    :return: dict with ``alpha`` (2-element adjustment speeds),
        ``beta`` (cointegrating vector [1, -beta2]),
        ``coefficients`` (lagged difference coefficients),
        ``residuals`` (n x 2 array), ``n``.
    :raises ValueError: On mismatched lengths or insufficient data.

    References
    ----------
    Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error
    correction: Representation, estimation, and testing. *Econometrica*,
    55(2), 251-276.

    Johansen, S. (1991). Estimation and hypothesis testing of
    cointegration vectors in Gaussian vector autoregressive models.
    *Econometrica*, 59(6), 1551-1580.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if len(y1) != len(y2):
        raise ValueError(f"y1 and y2 must have same length, got {len(y1)} and {len(y2)}.")
    n = len(y1)
    if n < lags + 3:
        raise ValueError(f"Need at least {lags + 3} observations, got {n}.")

    Y = np.column_stack([y1, y2])  # (n, 2)
    dY = np.diff(Y, axis=0)  # (n-1, 2)

    # Step 1: Estimate cointegrating vector via OLS (Engle-Granger)
    # y1 = beta0 + beta2 * y2 + u
    X_coint = np.column_stack([np.ones(n), y2])
    beta_coint = np.linalg.lstsq(X_coint, y1, rcond=None)[0]
    ect = y1 - X_coint @ beta_coint  # error correction term

    # Step 2: Build VECM design matrix
    # dY_t = alpha * ect_{t-1} + Gamma_1 * dY_{t-1} + ... + eps
    T = n - 1 - lags  # effective sample size
    if T < 3:
        raise ValueError(f"Effective sample too small ({T}) for {lags} lag(s).")

    # Dependent: dY[lags:, :]  (T x 2)
    dep = dY[lags:]

    # Regressors: ect_{t-1}, then lagged dY
    ect_lagged = ect[lags : n - 1]  # (T,)
    regressors = [ect_lagged.reshape(-1, 1)]
    for lag in range(1, lags + 1):
        regressors.append(dY[lags - lag : n - 1 - lag])
    X_reg = np.hstack(regressors)  # (T, 1 + 2*lags)

    # OLS for each equation
    coeffs = np.linalg.lstsq(X_reg, dep, rcond=None)[0]  # (1+2*lags, 2)
    alpha = coeffs[0]  # adjustment speeds (2,)
    gamma_coeffs = coeffs[1:]  # lagged difference coefficients

    residuals = dep - X_reg @ coeffs

    return {
        "alpha": alpha.tolist(),
        "beta": [1.0, float(-beta_coint[1])],
        "beta_intercept": float(beta_coint[0]),
        "coefficients": gamma_coeffs.tolist(),
        "residuals": residuals,
        "n": int(T),
    }


def cheatsheet() -> str:
    return "vecm({}) -> Vector Error Correction Model (VECM) for cointegrated series"
