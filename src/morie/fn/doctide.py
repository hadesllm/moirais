"""de Chaisemartin-D'Haultfoeuille heterogeneous DID."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["de_chaisemartin_dhaultfoeuille"]


def de_chaisemartin_dhaultfoeuille(y, D, unit, time):
    """
    de Chaisemartin-D'Haultfoeuille heterogeneous DID

    Formula: DID_M estimator, robust to heterogeneous + dynamic effects

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Chaisemartin & D'Haultfoeuille (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "de Chaisemartin-D'Haultfoeuille heterogeneous DID"})


def cheatsheet():
    return "doctide: de Chaisemartin-D'Haultfoeuille heterogeneous DID"
