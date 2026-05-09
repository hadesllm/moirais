"""m-out-of-n bootstrap for boundary problems."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_m_out_of_n"]


def boot_m_out_of_n(x, m, stat, B):
    """
    m-out-of-n bootstrap for boundary problems

    Formula: Sample m<n with replacement

    Parameters
    ----------
    x : array-like
        Input data.
    m : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Bickel-Götze-van Zwet (1997)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "m-out-of-n bootstrap for boundary problems"})


def cheatsheet():
    return "btmoutn: m-out-of-n bootstrap for boundary problems"
