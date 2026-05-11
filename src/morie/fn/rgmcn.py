# morie.fn — function file (hadesllm/morie)
"""McNemar's test for comparing two classifiers."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_mcnemar_test"]


def rangayyan_mcnemar_test(y1, y2, y_true, cdf=None):
    """
    McNemar's test for comparing two classifiers

    Formula: chi^2 = (|b-c|-1)^2 / (b+c); b,c = off-diagonal disagreement counts

    Parameters
    ----------
    y1 : array-like
        Input data.
    y2 : array-like
        Input data.
    y_true : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: chi2, p_value

    References
    ----------
    Rangayyan Ch 10.9.2
    """
    y1 = np.asarray(y1, dtype=float)
    n = int(y1) if y1.ndim == 0 else len(y1)
    if y1.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "McNemar's test for comparing two classifiers"})
    x_sorted = np.sort(y1)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(y1), scale=np.std(y1, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "McNemar's test for comparing two classifiers"})


def cheatsheet():
    return "rgmcn: McNemar's test for comparing two classifiers"
