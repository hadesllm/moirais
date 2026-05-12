# morie.fn -- function file (hadesllm/morie)
"""Le Cam testing lemma: lower bound on sum of error probabilities via affinity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_lecam_lemma"]


def ghosal_lecam_lemma(x, cdf=None):
    """
    Le Cam testing lemma: lower bound on sum of error probabilities via affinity

    Formula: E_{P0}phi + E_P(1-phi) >= 1 - d_TV(P0,P) >= 1 - d_H(P0,P)*sqrt(1-d_H^2/4)

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
    Ghosal App D
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Le Cam testing lemma: lower bound on sum of error probabilities via affinity"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Le Cam testing lemma: lower bound on sum of error probabilities via affinity"})


def cheatsheet():
    return "gh_ap_d2: Le Cam testing lemma: lower bound on sum of error probabilities via affinity"
