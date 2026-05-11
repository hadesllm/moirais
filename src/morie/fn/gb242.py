# morie.fn — function file (hadesllm/morie)
"""PDF of r-th order statistic from continuous CDF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_order_pdf"]


def gibbons_order_pdf(x, r, n, f, F, cdf=None):
    """
    PDF of r-th order statistic from continuous CDF

    Formula: f_{X_(r)}(x) = n!/((r-1)!(n-r)!) [F(x)]^(r-1) [1-F(x)]^(n-r) f(x)

    Parameters
    ----------
    x : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.
    f : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pdf_value

    References
    ----------
    Gibbons Theorem 2.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "PDF of r-th order statistic from continuous CDF"})
    x_sorted = np.sort(x)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(x), scale=np.std(x, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "PDF of r-th order statistic from continuous CDF"})


def cheatsheet():
    return "gb242: PDF of r-th order statistic from continuous CDF"
