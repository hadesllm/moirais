# morie.fn — function file (hadesllm/morie)
"""Ljung-Box portmanteau test for autocorrelation."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def ljung_box(residuals: np.ndarray, lags: int = 10, fitdf: int = 0, cdf=None) -> DescriptiveResult:
    """
    Ljung-Box test for autocorrelation in residuals.

    .. math::

        Q = n(n+2) \\sum_{k=1}^{h} \\frac{\\hat{\\rho}_k^2}{n-k}

    :param residuals: 1-D array of residuals.
    :param lags: Number of lags to test. Default 10.
    :param fitdf: Degrees of freedom used in model fitting. Default 0.
    :return: DescriptiveResult with Q statistic and p-value.
    :raises ValueError: If series too short.

    References
    ----------
    Ljung G.M. & Box G.E.P. (1978). On a measure of lack of fit in
    time series models. *Biometrika*, 65(2), 297-303.
    """
    e = np.asarray(residuals, dtype=float).ravel()
    n = len(e)
    if n < lags + 2:
        raise ValueError(f"Need at least {lags + 2} observations, got {n}.")
    e = e - e.mean()
    gamma0 = np.sum(e ** 2) / n
    if gamma0 < 1e-15:
        return DescriptiveResult(
            name="ljung_box", value=0.0,
            extra={"Q": 0.0, "p_value": 1.0, "lags": lags, "n": n},
        )
    Q = 0.0
    acf_vals = []
    for k in range(1, lags + 1):
        rho_k = np.sum(e[k:] * e[:-k]) / (n * gamma0)
        acf_vals.append(float(rho_k))
        Q += rho_k ** 2 / (n - k)
    Q *= n * (n + 2)
    df = max(lags - fitdf, 1)
    p_val = 1 - stats.chi2.cdf(Q, df)
    return DescriptiveResult(
        name="ljung_box",
        value=float(Q),
        extra={
            "Q": float(Q),
            "p_value": float(p_val),
            "df": df,
            "lags": lags,
            "acf": acf_vals,
            "n": n,
        },
    )


ljbxt = ljung_box


def cheatsheet() -> str:
    return "ljung_box({}) -> Ljung-Box portmanteau test."
