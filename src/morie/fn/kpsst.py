# morie.fn — function file (hadesllm/morie)
"""KPSS stationarity test."""

import numpy as np

from ._containers import DescriptiveResult


def kpss_test(y: np.ndarray, regression: str = "c", n_lags: int | None = None) -> DescriptiveResult:
    """
    KPSS test for stationarity.

    Tests H0: series is stationary vs H1: unit root.

    :param y: 1-D time series.
    :param regression: 'c' for level stationarity, 'ct' for trend. Default 'c'.
    :param n_lags: Bandwidth for long-run variance. Default int(sqrt(n)).
    :return: DescriptiveResult with KPSS statistic and critical values.
    :raises ValueError: If series too short.

    References
    ----------
    Kwiatkowski D., Phillips P.C.B., Schmidt P. & Shin Y. (1992).
    Testing the null hypothesis of stationarity. *J. Econometrics*,
    54, 159-178.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 10:
        raise ValueError(f"Need at least 10 observations, got {n}.")
    if n_lags is None:
        n_lags = int(np.sqrt(n))
    if regression == "ct":
        X = np.column_stack([np.ones(n), np.arange(1, n + 1)])
    else:
        X = np.ones((n, 1))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    cumsum = np.cumsum(resid)
    s2 = np.sum(resid ** 2) / n
    for lag in range(1, n_lags + 1):
        w = 1 - lag / (n_lags + 1)
        s2 += 2 * w * np.sum(resid[lag:] * resid[:-lag]) / n
    s2 = max(s2, 1e-10)
    stat = float(np.sum(cumsum ** 2) / (n ** 2 * s2))
    if regression == "ct":
        crit = {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216}
    else:
        crit = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    return DescriptiveResult(
        name="kpss_test",
        value=stat,
        extra={
            "statistic": stat,
            "critical_values": crit,
            "regression": regression,
            "n_lags": n_lags,
            "n": n,
        },
    )


kpsst = kpss_test


def cheatsheet() -> str:
    return "kpss_test({}) -> KPSS stationarity test."
