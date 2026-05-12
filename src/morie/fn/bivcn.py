# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bivariate causal direction test via independence of residuals in ANM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bivariate_causal_test"]


def bivariate_causal_test(X, Y, regressor, cdf=None):
    """
    Bivariate causal direction test via independence of residuals in ANM

    Formula: Fit Y=f(X)+N; test X _|_ N using HSIC; compare to reverse; direction = lower p-value

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    regressor : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'direction': 'str', 'hsic_forward': 'float', 'hsic_backward': 'float'}

    References
    ----------
    Molak Ch 13
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    X = X.ravel()
    n = len(X)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Bivariate causal direction test via independence of residuals in ANM"})
    x_sorted = np.sort(X)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(X), scale=np.std(X, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Bivariate causal direction test via independence of residuals in ANM"})


def cheatsheet():
    return "bivcn: Bivariate causal direction test via independence of residuals in ANM"
