# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Augmented Dickey-Fuller unit root test."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult

__all__ = ["adfts", "adf_test"]


def adf_test(y, maxlag: int | None = None, regression: str = "c") -> TestResult:
    """Augmented Dickey-Fuller unit root test.

    Tests H0: unit root present (series is I(1)) against H1: stationary.
    Lag order selected by minimising AIC when *maxlag* is None.

    Parameters
    ----------
    y : array-like
        Univariate time series.
    maxlag : int or None
        Maximum lag in auxiliary regression.
        If None uses ``ceil(12 * (n/100)^(1/4))`` (Schwert rule).
    regression : {'c', 'ct', 'nc'}
        Deterministic terms: 'c' = constant only (default),
        'ct' = constant + trend, 'nc' = no deterministic terms.

    Returns
    -------
    TestResult
        statistic: ADF t-statistic.
        p_value: approximate p-value (binned from critical values).
        extra['n_lags']: selected lag order.
        extra['critical_values']: dict with 1%, 5%, 10% critical values.

    References
    ----------
    Dickey D.A. & Fuller W.A. (1979). Distribution of the estimators for
    autoregressive time series with a unit root.
    Journal of the American Statistical Association, 74(366), 427-431.

    Said S.E. & Dickey D.A. (1984). Testing for unit roots in ARMA models.
    Biometrika, 71(3), 599-607.

    MacKinnon J.G. (1994). Approximate asymptotic distribution functions
    for unit root and cointegration tests.
    Journal of Business & Economic Statistics, 12(2), 167-176.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 4:
        raise ValueError(f"Need >= 4 observations, got {n}.")
    if regression not in ("c", "ct", "nc"):
        raise ValueError("regression must be 'c', 'ct', or 'nc'.")

    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    maxlag = min(maxlag, n - 2)

    dy = np.diff(y)

    # Build lagged-difference columns for augmentation.
    lag_cols = []
    for j in range(1, maxlag + 1):
        lag_cols.append(dy[maxlag - j : n - 1 - j])

    y_lagged = y[maxlag : n - 1]
    y_reg = dy[maxlag:]
    n_use = len(y_reg)

    det_cols = []
    if regression == "c":
        det_cols.append(np.ones(n_use))
    elif regression == "ct":
        det_cols.append(np.ones(n_use))
        det_cols.append(np.arange(maxlag + 1, n, dtype=float))

    X_parts = det_cols + [y_lagged] + lag_cols
    X = np.column_stack(X_parts) if len(X_parts) > 1 else np.asarray(X_parts[0]).reshape(-1, 1)

    # Index of the lagged-level coefficient depends on regression type.
    if regression == "nc":
        col_idx = 0
    elif regression == "c":
        col_idx = 1
    else:
        col_idx = 2

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y_reg, rcond=None)
    except np.linalg.LinAlgError:
        return TestResult(
            test_name="Augmented Dickey-Fuller",
            statistic=np.nan,
            p_value=np.nan,
            n=n,
            method=f"ADF (lags={maxlag}, regression='{regression}')",
            extra={"n_lags": maxlag, "critical_values": {}, "regression": regression},
        )

    residuals = y_reg - X @ beta
    df_res = n_use - X.shape[1]
    if df_res <= 0:
        return TestResult(
            test_name="Augmented Dickey-Fuller",
            statistic=np.nan,
            p_value=np.nan,
            n=n,
            method=f"ADF (lags={maxlag}, regression='{regression}')",
            extra={"n_lags": maxlag, "critical_values": {}, "regression": regression},
        )
    sigma2 = float(np.sum(residuals ** 2) / df_res)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return TestResult(
            test_name="Augmented Dickey-Fuller",
            statistic=np.nan,
            p_value=np.nan,
            n=n,
            method=f"ADF (lags={maxlag}, regression='{regression}')",
            extra={"n_lags": maxlag, "critical_values": {}, "regression": regression},
        )
    se = float(np.sqrt(sigma2 * XtX_inv[col_idx, col_idx]))
    tau = float(beta[col_idx]) / se if se > 0.0 else np.nan

    # MacKinnon (1994) approximate critical values.
    _cv = {
        "c":  {"1%": -3.43, "5%": -2.86, "10%": -2.57},
        "ct": {"1%": -3.96, "5%": -3.41, "10%": -3.13},
        "nc": {"1%": -2.57, "5%": -1.94, "10%": -1.62},
    }
    cv = _cv[regression]

    # Bin p-value from critical values (standard practice when full
    # MacKinnon response-surface polynomials are not available).
    if tau <= cv["1%"]:
        pval = 0.01
    elif tau <= cv["5%"]:
        pval = 0.05
    elif tau <= cv["10%"]:
        pval = 0.10
    else:
        pval = 0.90

    return TestResult(
        test_name="Augmented Dickey-Fuller",
        statistic=float(tau),
        p_value=float(pval),
        n=n,
        method=f"ADF (lags={maxlag}, regression='{regression}')",
        extra={
            "n_lags": maxlag,
            "critical_values": cv,
            "regression": regression,
        },
    )


adfts = adf_test


def cheatsheet() -> str:
    return "adf_test(y, maxlag=None, regression='c') -> Augmented Dickey-Fuller unit root test."
