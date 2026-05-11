"""Greatest Lower Bound (GLB) reliability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["greatest_lower_bound"]


def greatest_lower_bound(covariance, cdf=None):
    """
    Greatest Lower Bound (GLB) reliability

    Formula: GLB = 1 - sum tau^2_i / sigma_T^2 maximizing tau

    Parameters
    ----------
    covariance : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bentler (1972); Jackson-Agunwamba (1977)
    """
    covariance = np.asarray(covariance, dtype=float)
    n = len(covariance)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Greatest Lower Bound (GLB) reliability"})
    x_sorted = np.sort(covariance)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(covariance), scale=np.std(covariance, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Greatest Lower Bound (GLB) reliability"})


def cheatsheet():
    return "glb12r: Greatest Lower Bound (GLB) reliability"
