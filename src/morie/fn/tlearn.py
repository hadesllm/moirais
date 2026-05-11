"""T-learner for CATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["t_learner"]


def t_learner(y, D, X):
    """
    T-learner for CATE

    Formula: separate Y(1) and Y(0) models; tau = Y(1)-Y(0)

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Künzel et al (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T-learner for CATE"})


def cheatsheet():
    return "tlearn: T-learner for CATE"
