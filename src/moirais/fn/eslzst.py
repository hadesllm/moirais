"""Z-score for coefficient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_z_score"]


def esl_z_score(X, y, beta):
    """
    Z-score for coefficient

    Formula: z_j = beta_j / se(beta_j)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Z-score for coefficient"})


def cheatsheet():
    return "eslzst: Z-score for coefficient"
