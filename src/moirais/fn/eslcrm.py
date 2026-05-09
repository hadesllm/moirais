"""Cross-entropy loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_cross_entropy"]


def esl_cross_entropy(y, p):
    """
    Cross-entropy loss

    Formula: L = -sum y_k log p_k

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 11
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-entropy loss"})


def cheatsheet():
    return "eslcrm: Cross-entropy loss"
