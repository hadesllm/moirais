# moirais.fn — function file (hadesllm/moirais)
"""Phillips-Perron unit root test."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult

__all__ = ["pptst", "pp_test"]


def pp_test(y, lags: int | None = None) -> TestResult:
    """Phillips-Perron unit root test.

    Nonparametric correction for heteroskedasticity and autocorrelation in the
    residuals of the Dickey-Fuller regression.  Robust to unknown forms of
    serial correlation.

    Tests H0: unit root (series is I(1)).

    Parameters
    ----------
    y : array-like
        Univariate time series (n,).
    lags : int or None
        Lag truncation for Newey-West long-run variance estimator.
        If None uses ``ceil(4 * (n/100)^(2/9))``.

    Returns
    -------
    TestResult
        statistic: PP Z_tau test statistic.
        p_value: approximate p-value binned from MacKinnon (1994) critical
            values for regression with constant.
        extra['critical_values']: dict with 1%, 5%, 10% levels.
        extra['lags']: lag truncation used.
        extra['lrvar']: estimated long-run variance.

    References
    ----------
    Phillips P.C.B. & Perron P. (1988). Testing for a unit root in time series
    regression. Biometrika, 75(2), 335-346.

    MacKinnon J.G. (1994). Approximate asymptotic distribution functions for
    unit root and cointegration tests.
    Journal of Business & Economic Statistics, 12(2), 167-176.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 4:
        raise ValueError(f"Need >= 4 observations, got {n}.")

    if lags is None:
        lags = int(np.ceil(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 2))

    # OLS: delta(y) = mu + alpha*y[t-1] + epsilon.
    dy = np.diff(y)  # length n-1
    X = np.column_stack([np.ones(n - 1), y[: n - 1]])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    except np.linalg.LinAlgError:
        return TestResult(
            test_name="Phillips-Perron",
            statistic=np.nan,
            p_value=np.nan,
            n=n,
            method=f"PP Z_tau (lags={lags})",
            extra={"critical_values": {}, "lags": lags, "lrvar": np.nan},
        )

    resid = dy - X @ beta
    n_ols = n - 1
    df_res = n_ols - 2
    if df_res <= 0:
        return TestResult(
            test_name="Phillips-Perron",
            statistic=np.nan,
            p_value=np.nan,
            n=n,
            method=f"PP Z_tau (lags={lags})",
            extra={"critical_values": {}, "lags": lags, "lrvar": np.nan},
        )
    sigma2_ols = float(np.sum(resid ** 2) / df_res)

    # Newey-West long-run variance estimator.
    lrvar = float(np.mean(resid ** 2))
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1.0)
        acov = float(np.mean(resid[k:] * resid[: n_ols - k]))
        lrvar += 2.0 * weight * acov
    lrvar = max(lrvar, 1e-300)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return TestResult(
            test_name="Phillips-Perron",
            statistic=np.nan,
            p_value=np.nan,
            n=n,
            method=f"PP Z_tau (lags={lags})",
            extra={"critical_values": {}, "lags": lags, "lrvar": lrvar},
        )

    se_ols = float(np.sqrt(sigma2_ols * XtX_inv[1, 1]))
    tau_ols = float(beta[1]) / se_ols if se_ols > 0.0 else np.nan

    # Z_tau correction (Phillips & Perron 1988, eq. 23):
    # Z_tau = sqrt(sigma2_ols/lrvar)*tau - 0.5*(lrvar - sigma2_ols) * ...
    Sxx = float((X[:, 1] @ X[:, 1]) - n_ols * np.mean(X[:, 1]) ** 2)
    correction = 0.5 * (lrvar - sigma2_ols) * np.sqrt(n_ols) * np.sqrt(Sxx) / (
        lrvar * np.sqrt(float(np.sum(resid ** 2)))
    ) if lrvar > 0.0 and np.sum(resid ** 2) > 0.0 else 0.0

    z_tau = float(np.sqrt(sigma2_ols / lrvar)) * tau_ols - correction

    # MacKinnon (1994) critical values for regression with constant.
    cv = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    if z_tau <= cv["1%"]:
        pval = 0.01
    elif z_tau <= cv["5%"]:
        pval = 0.05
    elif z_tau <= cv["10%"]:
        pval = 0.10
    else:
        pval = 0.90

    return TestResult(
        test_name="Phillips-Perron",
        statistic=float(z_tau),
        p_value=float(pval),
        n=n,
        method=f"PP Z_tau (lags={lags})",
        extra={"critical_values": cv, "lags": lags, "lrvar": lrvar},
    )


pptst = pp_test


def cheatsheet() -> str:
    return "pp_test(y, lags=None) -> Phillips-Perron unit root test."
