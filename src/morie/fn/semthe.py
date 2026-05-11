"""SE of theta from test info."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sem_theta"]


def sem_theta(theta, items, cdf=None):
    """
    SE of theta from test info

    Formula: SE(theta_hat) = 1/sqrt(I(theta))

    Parameters
    ----------
    theta : array-like
        Input data.
    items : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lord (1980)
    """
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "SE of theta from test info"})
    x_sorted = np.sort(theta)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(theta), scale=np.std(theta, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "SE of theta from test info"})


def cheatsheet():
    return "semthe: SE of theta from test info"
