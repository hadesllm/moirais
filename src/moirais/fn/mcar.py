# moirais.fn — function file (hadesllm/moirais)
"""Little's MCAR test for missing completely at random."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ._richresult import RichResult


def littles_mcar_test(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> dict:
    r"""
    Little's chi-squared test for Missing Completely At Random (MCAR).

    Tests the null hypothesis that data are MCAR by comparing observed group
    means (grouped by missing-data pattern) against expected means under MCAR.

    The test statistic is:

    .. math::
        d^2 = \sum_j n_j (\bar{y}_j - \hat{\mu}_{obs,j})^T
              \hat{\Sigma}_{obs,j}^{-1} (\bar{y}_j - \hat{\mu}_{obs,j})

    where *j* indexes distinct missing-data patterns, :math:`\bar{y}_j` is the
    observed group mean for pattern *j*, and :math:`\hat{\mu}_{obs,j}`,
    :math:`\hat{\Sigma}_{obs,j}` are the EM estimates restricted to the
    observed variables in pattern *j*.  Under MCAR, :math:`d^2 \sim
    \chi^2(\sum_j p_j - p)` where :math:`p_j` is the number of observed
    variables in pattern *j* and *p* is the total number of variables.

    Only numeric columns are used.  Rows with no missing values or all missing
    values are included/excluded automatically.

    :param data: Input DataFrame (numeric columns, may contain NaN).
    :type data: pandas.DataFrame
    :param columns: Subset of columns to test.  Default: all numeric columns.
    :type columns: list[str] or None
    :return: Dictionary with ``chi2``, ``df``, ``p_value``.
    :rtype: dict
    :raises ValueError: If fewer than 2 columns or no missing values.

    References
    ----------
    Little, R. J. A. (1988). A test of missing completely at random for
    multivariate data with missing values. *Journal of the American
    Statistical Association*, 83(404), 1198--1202.
    https://doi.org/10.1080/01621459.1988.10478722
    """
    if columns is not None:
        data = data[columns]
    # Keep numeric only
    data = data.select_dtypes(include=[np.number]).copy()
    cols = data.columns.tolist()
    p = len(cols)

    if p < 2:
        raise ValueError("Need at least 2 numeric columns for MCAR test.")
    if not data.isna().any().any():
        raise ValueError("No missing values found; MCAR test is not applicable.")

    n = len(data)
    values = data.values  # n x p

    # EM estimates of mean and covariance (simple: use pairwise complete)
    mu_hat = np.nanmean(values, axis=0)
    # Pairwise complete covariance
    cov_hat = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(i, p):
            mask = ~(np.isnan(values[:, i]) | np.isnan(values[:, j]))
            if mask.sum() > 1:
                c = np.cov(values[mask, i], values[mask, j], ddof=1)
                cov_hat[i, j] = c[0, 1]
                cov_hat[j, i] = c[0, 1]
            else:
                cov_hat[i, j] = 0.0
                cov_hat[j, i] = 0.0
    np.fill_diagonal(cov_hat, np.nanvar(values, axis=0, ddof=1))
    # Replace any remaining NaN with 0
    cov_hat = np.nan_to_num(cov_hat, nan=0.0)

    # Group rows by missing pattern
    missing_indicator = np.isnan(values).astype(int)
    pattern_keys = [tuple(row) for row in missing_indicator]
    patterns: dict[tuple, list[int]] = {}
    for idx, key in enumerate(pattern_keys):
        patterns.setdefault(key, []).append(idx)

    chi2 = 0.0
    df = 0

    for pattern, indices in patterns.items():
        obs_vars = [k for k in range(p) if pattern[k] == 0]
        p_j = len(obs_vars)
        if p_j == 0:
            continue

        n_j = len(indices)
        subdata = values[np.ix_(indices, obs_vars)]
        y_bar = np.mean(subdata, axis=0)
        mu_obs = mu_hat[obs_vars]
        cov_obs = cov_hat[np.ix_(obs_vars, obs_vars)]

        diff = y_bar - mu_obs

        # Regularise covariance for invertibility
        cov_reg = cov_obs + np.eye(p_j) * 1e-10
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_reg)

        chi2 += float(n_j * diff @ cov_inv @ diff)
        df += p_j

    df -= p  # degrees of freedom = sum(p_j) - p

    if df <= 0:
        return RichResult(payload={"chi2": float(chi2), "df": 0, "p_value": float("nan")})

    # Chi-squared p-value using survival function approximation
    # Implemented without scipy: use regularised incomplete gamma
    p_value = _chi2_sf(chi2, df)

    return RichResult(payload={"chi2": float(chi2), "df": int(df), "p_value": float(p_value)})


def _chi2_sf(x: float, k: int) -> float:
    """Chi-squared survival function P(X > x) for X ~ chi2(k).

    Uses the incomplete gamma function via a series expansion.
    Falls back to scipy if available, otherwise uses pure-Python approx.
    """
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, k))
    except ImportError:
        pass
    # Pure-Python fallback: regularised upper incomplete gamma
    # P(X > x) = 1 - gammainc(k/2, x/2) where gammainc is the *lower*
    # regularised incomplete gamma function.
    return 1.0 - _lower_reg_gamma(k / 2.0, x / 2.0)


def _lower_reg_gamma(a: float, x: float) -> float:
    """Lower regularised incomplete gamma function P(a, x) via series."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    import math

    # Series expansion: P(a, x) = e^{-x} x^a sum_{n=0}^{inf} x^n / Gamma(a+n+1)
    # = e^{-x} x^a / Gamma(a) * sum_{n=0}^{inf} x^n / prod_{k=1}^{n}(a+k)
    # which simplifies to the series: term_n = term_{n-1} * x / (a + n)
    total = 1.0 / a
    term = 1.0 / a
    for n in range(1, 300):
        term *= x / (a + n)
        total += term
        if abs(term) < 1e-14 * abs(total):
            break
    log_val = a * math.log(x) - x - math.lgamma(a) + math.log(total)
    # Clamp to [0, 1]
    return min(max(math.exp(log_val), 0.0), 1.0)


mcar = littles_mcar_test


def cheatsheet() -> str:
    return "littles_mcar_test({}) -> Little's MCAR test for missing completely at random."
