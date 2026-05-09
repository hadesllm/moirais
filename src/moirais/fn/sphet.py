"""Spatial heterogeneity test (Breusch-Pagan spatial)."""

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def spatial_heterogeneity(residuals: np.ndarray, W: np.ndarray, cdf=None) -> TestResult:
    """
    Test for spatial heterogeneity using a Breusch-Pagan style test.

    Tests whether the variance of residuals depends on spatial location
    by regressing squared residuals on spatially lagged squared residuals.

    :param residuals: (n,) OLS residuals.
    :param W: (n, n) spatial weights matrix.
    :return: TestResult with chi-squared statistic and p-value.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Anselin L (1988). Spatial Econometrics: Methods and Models.
    Kluwer Academic Publishers.
    """
    e = np.asarray(residuals, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64)
    n = len(e)
    if W.shape != (n, n):
        raise ValueError("W must be (n, n).")
    e2 = e**2
    We2 = W @ e2
    X = np.column_stack([np.ones(n), We2])
    beta = np.linalg.lstsq(X, e2, rcond=None)[0]
    fitted = X @ beta
    ssr = np.sum((e2 - fitted) ** 2)
    sst = np.sum((e2 - e2.mean()) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else 0.0
    stat = float(n * r2)
    pval = float(1 - sp_stats.chi2.cdf(stat, df=1))
    return TestResult(
        test_name="spatial_heterogeneity",
        statistic=stat,
        p_value=pval,
        df=1.0,
        method="Breusch-Pagan spatial",
        n=n,
    )


sphet = spatial_heterogeneity


def cheatsheet() -> str:
    return "spatial_heterogeneity({}) -> Spatial heterogeneity test (Breusch-Pagan spatial)."
