# moirais.fn — function file (hadesllm/moirais)
"""Mardia multivariate normality test. 'All models are wrong, but some are useful. — George E. P. Box'"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def mardia_test(X: np.ndarray, cdf=None) -> TestResult:
    """Mardia's test for multivariate normality.

    Computes Mardia's multivariate skewness (b_{1,p}) and kurtosis
    (b_{2,p}) statistics and tests them against their asymptotic
    null distributions.

    :param X: (n, p) data matrix.
    :return: TestResult with skewness chi-squared statistic and kurtosis z.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    n, p = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    S = (Xc.T @ Xc) / n
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    D = Xc @ S_inv @ Xc.T

    b1p = float(np.sum(D**3) / (n**2))
    chi2_skew = n * b1p / 6.0
    df_skew = p * (p + 1) * (p + 2) / 6
    p_skew = float(1.0 - stats.chi2.cdf(chi2_skew, df_skew))

    b2p = float(np.trace(D**2) / n)
    z_kurt = (b2p - p * (p + 2)) / np.sqrt(8.0 * p * (p + 2) / n)
    p_kurt = float(2.0 * (1.0 - stats.norm.cdf(abs(z_kurt))))

    return TestResult(
        test_name="Mardia",
        statistic=chi2_skew,
        p_value=p_skew,
        df=df_skew,
        method="Mardia multivariate normality",
        n=n,
        extra={
            "skewness_b1p": b1p,
            "kurtosis_b2p": b2p,
            "kurtosis_z": float(z_kurt),
            "kurtosis_pvalue": p_kurt,
            "p": p,
        },
    )


mardm = mardia_test


def cheatsheet() -> str:
    return "Without music, life would be a mistake. — Friedrich Nietzsche"
