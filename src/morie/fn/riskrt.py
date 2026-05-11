"""Risk ratio (relative risk)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["risk_ratio"]


def risk_ratio(p_exposed, p_unexposed):
    """
    Risk ratio (relative risk)

    Formula: RR = p_e / p_u

    Parameters
    ----------
    p_exposed : array-like
        Input data.
    p_unexposed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rothman-Greenland (1998)
    """
    p_exposed = np.atleast_1d(np.asarray(p_exposed, dtype=float))
    n = len(p_exposed)
    result = float(np.mean(p_exposed))
    se = float(np.std(p_exposed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Risk ratio (relative risk)"})


def cheatsheet():
    return "riskrt: Risk ratio (relative risk)"
