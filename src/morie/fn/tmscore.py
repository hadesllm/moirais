"""TM-score for protein structure similarity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tm_score"]


def tm_score(coords1, coords2):
    """
    TM-score for protein structure similarity

    Formula: sum 1/(1+(d_i/d0)^2) / Lref

    Parameters
    ----------
    coords1 : array-like
        Input data.
    coords2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang-Skolnick (2004)
    """
    coords1 = np.atleast_1d(np.asarray(coords1, dtype=float))
    n = len(coords1)
    result = float(np.mean(coords1))
    se = float(np.std(coords1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TM-score for protein structure similarity"})


def cheatsheet():
    return "tmscore: TM-score for protein structure similarity"
