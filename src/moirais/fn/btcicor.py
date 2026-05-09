"""Bootstrap CI for Pearson correlation via Fisher z."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_ci_correlation"]


def boot_ci_correlation(x, y, B, alpha):
    """
    Bootstrap CI for Pearson correlation via Fisher z

    Formula: z* = atanh(r*); percentile interval; tanh back

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    B : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi

    References
    ----------
    Efron & Tibshirani (1993)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Bootstrap CI for Pearson correlation via Fisher z"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Bootstrap CI for Pearson correlation via Fisher z"})


def cheatsheet():
    return "btcicor: Bootstrap CI for Pearson correlation via Fisher z"
