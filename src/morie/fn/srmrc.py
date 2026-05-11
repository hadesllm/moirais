"""Spearman rank correlation with confidence intervals."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["srmrc"]


def srmrc(x, y, axis=0, ci_level=0.95, cdf=None):
    r"""
    Spearman rank correlation coefficient with confidence interval.

    Nonparametric measure of monotonic association.
    More robust than Pearson to outliers and nonlinearity.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    if y.ndim == 2:
        y = np.take(y, 0, axis=axis)

    n = len(x)
    if n < 3:
        raise ValueError("Sample size must be ≥3")

    # Rank transformation
    rank_x = sp_stats.rankdata(x)
    rank_y = sp_stats.rankdata(y)

    # Pearson correlation on ranks
    rho = np.corrcoef(rank_x, rank_y)[0, 1]

    # z-transformation: z = 0.5 * ln((1+ρ)/(1-ρ))
    rho_clipped = np.clip(rho, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + rho_clipped) / (1 - rho_clipped))

    # Standard error of z
    se_z = 1 / np.sqrt(n - 3)

    # Critical value for CI
    alpha = 1 - ci_level
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)

    # CI on z scale
    z_lower = z - z_crit * se_z
    z_upper = z + z_crit * se_z

    # Transform back to ρ scale
    rho_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    rho_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    # p-value
    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2 + 1e-10))
    p_value = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), n - 2))

    return {
        "correlation": float(rho),
        "p_value": float(p_value),
        "ci_lower": float(rho_lower),
        "ci_upper": float(rho_upper),
        "n": int(n),
    }
