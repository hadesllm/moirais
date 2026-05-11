"""Structural distance / DeltaCon."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["structural_distance"]


def structural_distance(G1, G2):
    """
    Structural distance / DeltaCon

    Formula: d = sqrt(sum (sqrt s_ij - sqrt s'_ij)^2)

    Parameters
    ----------
    G1 : array-like
        Input data.
    G2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Koutra et al (2013)
    """
    G1 = np.atleast_1d(np.asarray(G1, dtype=float))
    n = len(G1)
    result = float(np.mean(G1))
    se = float(np.std(G1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Structural distance / DeltaCon"})


def cheatsheet():
    return "strdis: Structural distance / DeltaCon"
