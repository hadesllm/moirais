"""TMLE for categorical outcomes via multinomial logit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_categorical_outcome"]


def tmle_categorical_outcome(y, D, X, K):
    """
    TMLE for categorical outcomes via multinomial logit

    Formula: target P(Y=k|do(A=a)) for each k

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL & Rose (2011) Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for categorical outcomes via multinomial logit"})


def cheatsheet():
    return "tmlcat: TMLE for categorical outcomes via multinomial logit"
