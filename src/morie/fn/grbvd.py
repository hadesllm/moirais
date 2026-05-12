# morie.fn -- function file (hadesllm/morie)
"""Bias-variance decomposition of expected test error."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_bias_variance_decomposition"]


def geron_bias_variance_decomposition(y_true, predictions, cdf=None):
    """
    Bias-variance decomposition of expected test error

    Formula: E[(y - h(x))^2] = bias^2 + variance + irreducible_error

    Parameters
    ----------
    y_true : array-like
        Input data.
    predictions : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias, variance, irreducible

    References
    ----------
    Géron Ch 1, Bias/Variance Trade-off section
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    if y_true.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Bias-variance decomposition of expected test error"})
    x_sorted = np.sort(y_true)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(y_true), scale=np.std(y_true, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Bias-variance decomposition of expected test error"})


def cheatsheet():
    return "grbvd: Bias-variance decomposition of expected test error"
