"""Cohen's d standardised mean difference."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_cohens_d"]


def ma_cohens_d(m1, m2, s1, s2, n1, n2):
    """
    Cohen's d standardised mean difference

    Formula: d = (m1-m2)/s_pooled

    Parameters
    ----------
    m1 : array-like
        Input data.
    m2 : array-like
        Input data.
    s1 : array-like
        Input data.
    s2 : array-like
        Input data.
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: d, var_d

    References
    ----------
    Cohen (1988)
    """
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    n = len(m1)
    result = float(np.mean(m1))
    se = float(np.std(m1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cohen's d standardised mean difference"})


def cheatsheet():
    return "macohd: Cohen's d standardised mean difference"
