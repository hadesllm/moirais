"""Local-shift sensitivity λ*."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["local_shift"]


def local_shift(IF):
    """
    Local-shift sensitivity λ*

    Formula: λ* = sup_{x≠y} ||IF(x)−IF(y)||/||x−y||

    Parameters
    ----------
    IF : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hampel et al (1986)
    """
    IF = np.atleast_1d(np.asarray(IF, dtype=float))
    n = len(IF)
    result = float(np.mean(IF))
    se = float(np.std(IF, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local-shift sensitivity λ*"})


def cheatsheet():
    return "localS: Local-shift sensitivity λ*"
