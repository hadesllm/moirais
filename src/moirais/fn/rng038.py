"""Synthetic test signal: sum of a sine and a cosine.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_test_signal_sin_cos"]


def rangayyan_ch3_test_signal_sin_cos(t, cdf=None):
    """
    Synthetic test signal: sum of a sine and a cosine.

    Formula: x(t) = 5 sin(2*pi*2*t) + 2 cos(2*pi*3*t)

    Parameters
    ----------
    t : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.40, p. 112
    """
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Synthetic test signal: sum of a sine and a cosine."})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Synthetic test signal: sum of a sine and a cosine."})


def cheatsheet():
    return "rng038: Synthetic test signal: sum of a sine and a cosine."
