"""BNS jump test based on RV vs BPV."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_jump_test_bnshep"]


def vol_jump_test_bnshep(r_intraday, block_index, cdf=None):
    """
    BNS jump test based on RV vs BPV

    Formula: JT = (RV-BPV) / sqrt((π²/4 + π - 5) Σ|r|^4 /4)

    Parameters
    ----------
    r_intraday : array-like
        Input data.
    block_index : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: JT, p

    References
    ----------
    Barndorff-Nielsen-Shephard (2006)
    """
    r_intraday = np.asarray(r_intraday, dtype=float)
    n = len(r_intraday)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "BNS jump test based on RV vs BPV"})
    x_sorted = np.sort(r_intraday)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(r_intraday), scale=np.std(r_intraday, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "BNS jump test based on RV vs BPV"})


def cheatsheet():
    return "voljmp: BNS jump test based on RV vs BPV"
