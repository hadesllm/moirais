"""Composite test signal expressed in terms of three delayed scaled copies of g(n).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_composite_signal_in_terms_of_g"]


def rangayyan_ch4_composite_signal_in_terms_of_g(g, n, cdf=None):
    """
    Composite test signal expressed in terms of three delayed scaled copies of g(n).

    Formula: x(n) = g(n-5) + 0.5*g(n-16) + 0.25*g(n-26)

    Parameters
    ----------
    g : array-like
        Input data.
    n : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.53, p. 240
    """
    g = np.asarray(g, dtype=float)
    n = len(g)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Composite test signal expressed in terms of three delayed scaled copies of g(n)."})
    x_sorted = np.sort(g)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(g), scale=np.std(g, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Composite test signal expressed in terms of three delayed scaled copies of g(n)."})


def cheatsheet():
    return "rng225: Composite test signal expressed in terms of three delayed scaled copies of g(n)."
