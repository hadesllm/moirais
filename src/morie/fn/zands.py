"""Zivot-Andrews structural break unit root test."""

import numpy as np

from ._containers import DescriptiveResult


def za_test(y: np.ndarray, trim: float = 0.15) -> DescriptiveResult:
    """
    Zivot-Andrews test for a unit root with a structural break.

    Searches over all possible break dates and returns the minimum
    t-statistic (most negative = strongest evidence against unit root).

    :param y: 1-D time series.
    :param trim: Fraction of endpoints to exclude. Default 0.15.
    :return: DescriptiveResult with min t-stat and break date index.
    :raises ValueError: If series too short.

    References
    ----------
    Zivot E. & Andrews D.W.K. (1992). Further evidence on the Great
    Crash, the oil-price shock, and the unit-root hypothesis. *JBES*,
    10(3), 251-270.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")
    start = int(n * trim)
    end = int(n * (1 - trim))
    if start >= end:
        raise ValueError("Trim too large for this sample size.")
    dy = np.diff(y)
    best_t = np.inf
    best_break = start
    for bp in range(start, end):
        T = n - 1
        dep = dy
        DU = np.zeros(T)
        DU[bp:] = 1.0
        DT = np.zeros(T)
        DT[bp:] = np.arange(1, T - bp + 1)
        X = np.column_stack([np.ones(T), y[:-1], np.arange(1, T + 1), DU, DT])
        beta = np.linalg.lstsq(X, dep, rcond=None)[0]
        resid = dep - X @ beta
        sig2 = float(np.sum(resid ** 2) / (T - X.shape[1]))
        se = float(np.sqrt(max(sig2 * np.linalg.inv(X.T @ X)[1, 1], 1e-20)))
        t_stat = float(beta[1] / se)
        if t_stat < best_t:
            best_t = t_stat
            best_break = bp
    crit = {"1%": -5.34, "5%": -4.80, "10%": -4.58}
    return DescriptiveResult(
        name="za_test",
        value=best_t,
        extra={
            "statistic": best_t,
            "break_index": best_break,
            "critical_values": crit,
            "trim": trim,
            "n": n,
        },
    )


zands = za_test


def cheatsheet() -> str:
    return "za_test({}) -> Zivot-Andrews structural break unit root test."
