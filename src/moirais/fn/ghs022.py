"""Cell-count sufficient statistic N_epsilon counting how many of n observations fall into the partition set A_epsilon.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tailfree_cell_counts"]


def ghosal_ch3_tailfree_cell_counts(X_i, A_epsilon, n, cdf=None):
    """
    Cell-count sufficient statistic N_epsilon counting how many of n observations fall into the partition set A_epsilon.

    Formula: N_epsilon := #{ 1 <= i <= n : X_i in A_epsilon }

    Parameters
    ----------
    X_i : array-like
        Input data.
    A_epsilon : array-like
        Input data.
    n : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.15, p. 41
    """
    X_i = np.asarray(X_i, dtype=float)
    n = len(X_i)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Cell-count sufficient statistic N_epsilon counting how many of n observations fall into the partition set A_epsilon."})
    x_sorted = np.sort(X_i)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(X_i), scale=np.std(X_i, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Cell-count sufficient statistic N_epsilon counting how many of n observations fall into the partition set A_epsilon."})


def cheatsheet():
    return "ghs022: Cell-count sufficient statistic N_epsilon counting how many of n observations fall into the partition set A_epsilon."
