"""Spatial regimes (Chow test for spatial heterogeneity)."""

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def spatial_regime(y: np.ndarray, X: np.ndarray, regions: np.ndarray, cdf=None) -> DescriptiveResult:
    """
    Spatial regimes via Chow test for structural breaks across regions.

    Fits pooled and region-specific OLS models and tests parameter
    stability across spatial regimes using an F-test.

    :param y: (n,) dependent variable.
    :param X: (n, k) explanatory variables.
    :param regions: (n,) region labels.
    :return: DescriptiveResult with F-statistic and regime coefficients.

    References
    ----------
    Anselin L (1990). Spatial Dependence and Spatial Structural
    Instability in Applied Regression Analysis.
    Journal of Regional Science, 30(2), 185-207.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    regions = np.asarray(regions)
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    k = X.shape[1]
    beta_pool = np.linalg.lstsq(X, y, rcond=None)[0]
    rss_pool = float(np.sum((y - X @ beta_pool) ** 2))
    unique_regions = np.unique(regions)
    J = len(unique_regions)
    rss_regime = 0.0
    regime_coefs = {}
    for r in unique_regions:
        mask = regions == r
        yr, Xr = y[mask], X[mask]
        br = np.linalg.lstsq(Xr, yr, rcond=None)[0]
        rss_regime += float(np.sum((yr - Xr @ br) ** 2))
        regime_coefs[str(r)] = br.tolist()
    df1 = (J - 1) * k
    df2 = n - J * k
    f_stat = ((rss_pool - rss_regime) / df1) / (rss_regime / df2) if df2 > 0 and df1 > 0 else 0.0
    p_val = float(1 - sp_stats.f.cdf(f_stat, df1, df2)) if df2 > 0 else 1.0
    return DescriptiveResult(
        name="spatial_regime",
        value=float(f_stat),
        extra={
            "f_statistic": f_stat,
            "p_value": p_val,
            "df1": df1,
            "df2": df2,
            "regime_coefficients": regime_coefs,
            "n_regimes": J,
            "n": n,
        },
    )


spreg = spatial_regime


def cheatsheet() -> str:
    return "spatial_regime({}) -> Spatial regimes (Chow test for spatial heterogeneity)."
