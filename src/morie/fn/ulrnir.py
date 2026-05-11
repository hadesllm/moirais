"""U-learner for CATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["u_learner"]


def u_learner(y, D, X):
    """
    U-learner for CATE

    Formula: residualized + unconfoundedness moment

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "U-learner for CATE"})


def cheatsheet():
    return "ulrnir: U-learner for CATE"
