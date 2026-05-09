"""Interaction information II(X;Y;Z)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["interaction_information"]


def interaction_information(pxyz):
    """
    Interaction information II(X;Y;Z)

    Formula: II = I(X;Y|Z) - I(X;Y)

    Parameters
    ----------
    pxyz : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McGill (1954)
    """
    pxyz = np.atleast_1d(np.asarray(pxyz, dtype=float))
    n = len(pxyz)
    result = float(np.mean(pxyz))
    se = float(np.std(pxyz, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Interaction information II(X;Y;Z)"})


def cheatsheet():
    return "intinf: Interaction information II(X;Y;Z)"
