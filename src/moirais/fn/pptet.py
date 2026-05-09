# moirais.fn — function file (hadesllm/moirais)
"""Phillips-Perron unit root test."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def pp_test(y: np.ndarray, n_lags: int | None = None, cdf=None) -> DescriptiveResult:
    """
    Phillips-Perron test for a unit root.

    Non-parametric correction to the Dickey-Fuller statistic for
    serial correlation and heteroscedasticity.

    :param y: 1-D time series.
    :param n_lags: Bandwidth for Newey-West. Default int(4*(n/100)^{2/9}).
    :return: DescriptiveResult with PP statistic and approximate p-value.
    :raises ValueError: If series too short.

    References
    ----------
    Phillips P.C.B. & Perron P. (1988). Testing for a unit root in
    time series regression. *Biometrika*, 75(2), 335-346.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 10:
        raise ValueError(f"Need at least 10 observations, got {n}.")
    if n_lags is None:
        n_lags = int(4 * (n / 100) ** (2 / 9))
    n_lags = max(1, n_lags)
    X = np.column_stack([np.ones(n - 1), y[:-1]])
    dep = y[1:]
    beta = np.linalg.lstsq(X, dep, rcond=None)[0]
    resid = dep - X @ beta
    T = n - 1
    sig2 = float(np.sum(resid ** 2) / T)
    lam2 = sig2
    for lag in range(1, n_lags + 1):
        w = 1 - lag / (n_lags + 1)
        lam2 += 2 * w * float(np.sum(resid[lag:] * resid[:-lag]) / T)
    lam2 = max(lam2, 1e-10)
    rho_hat = beta[1]
    XtX = X.T @ X
    se_rho = float(np.sqrt(max(sig2 * np.linalg.inv(XtX)[1, 1], 1e-20)))
    t_alpha = (rho_hat - 1) / se_rho
    pp_stat = float(
        T * (rho_hat - 1) / np.sqrt(lam2 / sig2)
        - 0.5 * (lam2 - sig2) * np.sqrt(T) / (se_rho * np.sqrt(lam2))
    )
    crit_1 = -3.43
    crit_5 = -2.86
    crit_10 = -2.57
    if pp_stat < crit_1:
        approx_p = 0.005
    elif pp_stat < crit_5:
        approx_p = 0.03
    elif pp_stat < crit_10:
        approx_p = 0.07
    else:
        approx_p = min(1.0, stats.norm.cdf(pp_stat) * 2)
    return DescriptiveResult(
        name="pp_test",
        value=pp_stat,
        extra={
            "statistic": pp_stat,
            "p_value": approx_p,
            "t_alpha": float(t_alpha),
            "critical_values": {"1%": crit_1, "5%": crit_5, "10%": crit_10},
            "n_lags": n_lags,
            "n": n,
        },
    )


pptet = pp_test


def cheatsheet() -> str:
    return "pp_test({}) -> Phillips-Perron unit root test."
