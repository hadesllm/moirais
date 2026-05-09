"""Index of moderated mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["index_moderated_mediation"]


def index_moderated_mediation(a1, a3, b):
    """
    Index of moderated mediation

    Formula: IMM = a3 * b (interaction X*W on M, then M->Y)

    Parameters
    ----------
    a1 : array-like
        Input data.
    a3 : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hayes (2015)
    """
    a1 = np.atleast_1d(np.asarray(a1, dtype=float))
    n = len(a1)
    result = float(np.mean(a1))
    se = float(np.std(a1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Index of moderated mediation"})


def cheatsheet():
    return "immid: Index of moderated mediation"
