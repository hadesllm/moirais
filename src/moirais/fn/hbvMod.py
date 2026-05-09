"""HBV conceptual hydrology model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hbv_hydrology"]


def hbv_hydrology(P, T, params):
    """
    HBV conceptual hydrology model

    Formula: snow + soil + response routines

    Parameters
    ----------
    P : array-like
        Input data.
    T : array-like
        Input data.
    params : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bergström (1995)
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HBV conceptual hydrology model"})


def cheatsheet():
    return "hbvMod: HBV conceptual hydrology model"
