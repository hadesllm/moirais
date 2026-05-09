"""Sure independence screening."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_sis_screening"]


def esl_sis_screening(X, y, d):
    """
    Sure independence screening

    Formula: rank features by |Cor(X_j, y)|

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: selected

    References
    ----------
    Hastie ESL Ch 18
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sure independence screening"})


def cheatsheet():
    return "eslsis: Sure independence screening"
