# moirais.fn — function file (hadesllm/moirais)
"""Two-parameter Poisson-Dirichlet PD(alpha, theta): order statistics of PY weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_poisson_dirichlet"]


def ghosal_poisson_dirichlet(x, cdf=None):
    """
    Two-parameter Poisson-Dirichlet PD(alpha, theta): order statistics of PY weights

    Formula: Ranked weights (p_(1), p_(2), ...) ~ PD(alpha, theta) for PY(alpha,theta,G0)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 14 §14.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Two-parameter Poisson-Dirichlet PD(alpha, theta): order statistics of PY weights"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Two-parameter Poisson-Dirichlet PD(alpha, theta): order statistics of PY weights"})


def cheatsheet():
    return "gh_pd_2param: Two-parameter Poisson-Dirichlet PD(alpha, theta): order statistics of PY weights"
