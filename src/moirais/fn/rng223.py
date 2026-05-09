"""Synthetic test signal composed of three occurrences of a basic pattern (impulses).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_test_signal_three_events"]


def rangayyan_ch4_test_signal_three_events(n, cdf=None):
    """
    Synthetic test signal composed of three occurrences of a basic pattern (impulses).

    Formula: x(n) = 3*delta(n-5) + 2*delta(n-6) + delta(n-7) + 1.5*delta(n-16) + delta(n-17) + 0.5*delta(n-18) + 0.75*delta(n-26) + 0.5*delta(n-27) + 0.25*delta(n-28)

    Parameters
    ----------
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
    Rangayyan (2024), Ch 4, Eq 4.51, p. 240
    """
    n = np.asarray(n, dtype=float)
    n = len(n)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Synthetic test signal composed of three occurrences of a basic pattern (impulses)."})
    x_sorted = np.sort(n)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(n), scale=np.std(n, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Synthetic test signal composed of three occurrences of a basic pattern (impulses)."})


def cheatsheet():
    return "rng223: Synthetic test signal composed of three occurrences of a basic pattern (impulses)."
