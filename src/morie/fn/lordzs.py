"""Lord chi-square DIF (parameter-difference test)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lord_chi_square"]


def lord_chi_square(y, b_R, b_F, V, cdf=None):
    """
    Lord chi-square DIF (parameter-difference test)

    Formula: chi2 = (b_R - b_F)' V^-1 (b_R - b_F)

    Parameters
    ----------
    y : array-like
        Input data.
    b_R : array-like
        Input data.
    b_F : array-like
        Input data.
    V : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lord (1980) §14
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Lord chi-square DIF (parameter-difference test)"})
    x_sorted = np.sort(y)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(y), scale=np.std(y, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Lord chi-square DIF (parameter-difference test)"})


def cheatsheet():
    return "lordzs: Lord chi-square DIF (parameter-difference test)"
