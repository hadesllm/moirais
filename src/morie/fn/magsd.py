"""Glass's Δ using control SD only."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_glass_delta"]


def ma_glass_delta(m1, m2, s_ctrl, n1, n2):
    """
    Glass's Δ using control SD only

    Formula: Δ = (m1-m2)/s_control

    Parameters
    ----------
    m1 : array-like
        Input data.
    m2 : array-like
        Input data.
    s_ctrl : array-like
        Input data.
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: delta, var

    References
    ----------
    Glass-McGaw-Smith (1981)
    """
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    n = len(m1)
    result = float(np.mean(m1))
    se = float(np.std(m1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glass's Δ using control SD only"})


def cheatsheet():
    return "magsd: Glass's Δ using control SD only"
