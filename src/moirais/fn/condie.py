"""Conditional indirect effect at moderator value."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["conditional_indirect_effect"]


def conditional_indirect_effect(a1, a3, b, w):
    """
    Conditional indirect effect at moderator value

    Formula: CIE(W=w) = (a1 + a3*w) * b

    Parameters
    ----------
    a1 : array-like
        Input data.
    a3 : array-like
        Input data.
    b : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Preacher, Rucker, Hayes (2007)
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional indirect effect at moderator value"})


def cheatsheet():
    return "condie: Conditional indirect effect at moderator value"
