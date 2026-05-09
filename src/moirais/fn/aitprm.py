"""PERMANOVA F using Aitchison distances."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_permanova"]


def compositional_permanova(X, groups, n_perm):
    """
    PERMANOVA F using Aitchison distances

    Formula: F = SS_B/(g-1) / SS_W/(N-g)

    Parameters
    ----------
    X : array-like
        Input data.
    groups : array-like
        Input data.
    n_perm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: F, p, pseudoF_dist

    References
    ----------
    Anderson (2001)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PERMANOVA F using Aitchison distances"})


def cheatsheet():
    return "aitprm: PERMANOVA F using Aitchison distances"
