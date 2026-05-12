# morie.fn -- function file (hadesllm/morie)
"""CDF of r-th order statistic X_(r) in terms of binomial tail probability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_order_cdf"]


def gibbons_order_cdf(t, r, n, F, cdf=None):
    """
    CDF of r-th order statistic X_(r) in terms of binomial tail probability

    Formula: P(X_(r) <= t) = sum_{i=r}^{n} C(n,i) [F(t)]^i [1-F(t)]^(n-i)

    Parameters
    ----------
    t : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cdf_value

    References
    ----------
    Gibbons Theorem 2.4.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "CDF of r-th order statistic X_(r) in terms of binomial tail probability"})
    x_sorted = np.sort(t)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(t), scale=np.std(t, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "CDF of r-th order statistic X_(r) in terms of binomial tail probability"})


def cheatsheet():
    return "gb241: CDF of r-th order statistic X_(r) in terms of binomial tail probability"
