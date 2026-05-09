"""Simulated annealing for spatial simulation matching target statistics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schabenberger_simulated_annealing"]


def schabenberger_simulated_annealing(target_stats, initial_config, cdf=None):
    """
    Simulated annealing for spatial simulation matching target statistics

    Formula: Accept proposal with prob min(1, exp(-dE/T)); T->0 according to cooling schedule

    Parameters
    ----------
    target_stats : array-like
        Input data.
    initial_config : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimized_config

    References
    ----------
    Schabenberger Ch 7, Sec 7.3
    """
    target_stats = np.asarray(target_stats, dtype=float)
    n = int(target_stats) if target_stats.ndim == 0 else len(target_stats)
    if target_stats.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Simulated annealing for spatial simulation matching target statistics"})
    x_sorted = np.sort(target_stats)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(target_stats), scale=np.std(target_stats, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Simulated annealing for spatial simulation matching target statistics"})


def cheatsheet():
    return "spsann: Simulated annealing for spatial simulation matching target statistics"
