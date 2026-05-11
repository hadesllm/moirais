"""E-value for unmeasured confounding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evalue"]


def evalue(RR):
    """
    E-value for unmeasured confounding

    Formula: E = RR + √(RR(RR−1)) for RR>1

    Parameters
    ----------
    RR : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele-Ding (2017)
    """
    RR = np.atleast_1d(np.asarray(RR, dtype=float))
    n = len(RR)
    result = float(np.mean(RR))
    se = float(np.std(RR, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "E-value for unmeasured confounding"})


def cheatsheet():
    return "evalu: E-value for unmeasured confounding"
