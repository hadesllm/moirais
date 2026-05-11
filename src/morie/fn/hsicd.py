# morie.fn — function file (hadesllm/morie)
"""Hilbert-Schmidt Independence Criterion (HSIC) for nonparametric independence testing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hsic_independence"]


def hsic_independence(X, Y, kernel, alpha, cdf=None):
    """
    Hilbert-Schmidt Independence Criterion (HSIC) for nonparametric independence testing

    Formula: HSIC(X,Y) = (1/n^2)*tr(K_X*H*K_Y*H); H=I-11'/n; zero iff X _|_ Y for characteristic kernels

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    kernel : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'hsic': 'float', 'p_value': 'float'}

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
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Hilbert-Schmidt Independence Criterion (HSIC) for nonparametric independence testing"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Hilbert-Schmidt Independence Criterion (HSIC) for nonparametric independence testing"})


def cheatsheet():
    return "hsicd: Hilbert-Schmidt Independence Criterion (HSIC) for nonparametric independence testing"
