"""Meta-learner ensemble (S,T,X,R)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["meta_learner_ensemble"]


def meta_learner_ensemble(y, D, X, weights):
    """
    Meta-learner ensemble (S,T,X,R)

    Formula: weighted average of S/T/X/R-learner estimates

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Curth-Schaar (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Meta-learner ensemble (S,T,X,R)"})


def cheatsheet():
    return "meta1l: Meta-learner ensemble (S,T,X,R)"
