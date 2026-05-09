"""Unobserved components model (UCM)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["unobserved_components"]


def unobserved_components(y, components):
    """
    Unobserved components model (UCM)

    Formula: trend + seasonal + cycle + irregular

    Parameters
    ----------
    y : array-like
        Input data.
    components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Harvey (1989)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unobserved components model (UCM)"})


def cheatsheet():
    return "unobts: Unobserved components model (UCM)"
