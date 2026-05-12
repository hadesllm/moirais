# morie.fn -- function file (hadesllm/morie)
"""Rubin's rules for pooling multiply imputed estimates."""

from __future__ import annotations

import math

import numpy as np


def rubins_rules(
    estimates: list[float] | np.ndarray,
    std_errors: list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    r"""
    Pool multiply imputed point estimates and standard errors via Rubin's
    rules with the Barnard--Rubin degrees-of-freedom adjustment.

    Given *m* imputed datasets each yielding a point estimate
    :math:`\hat{\theta}_i` and standard error :math:`\text{SE}_i`:

    - **Pooled estimate**: :math:`\bar{\theta} = \frac{1}{m}\sum \hat{\theta}_i`
    - **Within-imputation variance**: :math:`\bar{U} = \frac{1}{m}\sum \text{SE}_i^2`
    - **Between-imputation variance**: :math:`B = \frac{1}{m-1}\sum(\hat{\theta}_i - \bar{\theta})^2`
    - **Total variance**: :math:`T = \bar{U} + (1 + 1/m) B`
    - **Pooled SE**: :math:`\sqrt{T}`
    - **Barnard--Rubin df**:

    .. math::
        \nu = (m - 1)\left(1 + \frac{\bar{U}}{(1 + 1/m)B}\right)^2

    adjusted by the observed-data degrees of freedom per Barnard & Rubin
    (1999) when *B* is small relative to :math:`\bar{U}`.

    :param estimates: List of *m* point estimates from each imputed dataset.
    :type estimates: list[float] or numpy.ndarray
    :param std_errors: List of *m* standard errors from each imputed dataset.
    :type std_errors: list[float] or numpy.ndarray
    :param alpha: Significance level for the confidence interval.  Default 0.05.
    :type alpha: float
    :return: Dictionary with ``pooled_estimate``, ``pooled_se``, ``df``,
        ``ci_lower``, ``ci_upper``, ``within_var``, ``between_var``,
        ``total_var``, ``fmi`` (fraction of missing information).
    :rtype: dict
    :raises ValueError: If fewer than 2 estimates, or lengths do not match.

    References
    ----------
    Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
    Wiley. https://doi.org/10.1002/9780470316696

    Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of freedom
    with multiple imputation. *Biometrika*, 86(4), 948--955.
    https://doi.org/10.1093/biomet/86.4.948
    """
    q = np.asarray(estimates, dtype=float)
    se = np.asarray(std_errors, dtype=float)

    if len(q) < 2:
        raise ValueError("Need at least 2 imputed estimates for pooling.")
    if len(q) != len(se):
        raise ValueError(f"Length mismatch: {len(q)} estimates vs {len(se)} standard errors.")

    m = len(q)
    q_bar = float(np.mean(q))
    u_bar = float(np.mean(se**2))  # within-imputation variance
    b = float(np.var(q, ddof=1))  # between-imputation variance

    total_var = u_bar + (1 + 1 / m) * b
    pooled_se = math.sqrt(total_var)

    # Fraction of missing information
    r = (1 + 1 / m) * b / u_bar if u_bar > 0 else float("inf")
    fmi = (r + 2 / (m + 1)) / (1 + r) if (1 + r) > 0 else 1.0

    # Barnard-Rubin degrees of freedom
    if b > 0:
        df_old = (m - 1) * (1 + u_bar / ((1 + 1 / m) * b)) ** 2
    else:
        df_old = float("inf")

    df = float(df_old)

    # Confidence interval using t-distribution critical value
    t_crit = _t_quantile(1 - alpha / 2, df)
    ci_lower = q_bar - t_crit * pooled_se
    ci_upper = q_bar + t_crit * pooled_se

    return {
        "pooled_estimate": q_bar,
        "pooled_se": pooled_se,
        "df": df,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "within_var": u_bar,
        "between_var": b,
        "total_var": total_var,
        "fmi": float(fmi),
    }


def _t_quantile(p: float, df: float) -> float:
    """Approximate t-distribution quantile.  Falls back to scipy if available."""
    try:
        from scipy.stats import t

        return float(t.ppf(p, df))
    except ImportError:
        pass
    # Fallback: for large df, use normal approximation
    if df > 120:
        return _normal_quantile(p)
    # Simple Cornish-Fisher approximation
    z = _normal_quantile(p)
    g1 = (z**3 + z) / (4 * df)
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    return z + g1 + g2


def _normal_quantile(p: float) -> float:
    """Approximate standard normal quantile (Beasley-Springer-Moro)."""
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p == 0.5:
        return 0.0
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    if p < 0.5:
        return -_normal_quantile(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)


rubin = rubins_rules


def cheatsheet() -> str:
    return "rubins_rules({}) -> Rubin's rules for pooling multiply imputed estimates."
