"""BIC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_bic_score"]


def esl_bic_score(loglik, d, N):
    """
    BIC

    Formula: BIC = -2 log L + d log N

    Parameters
    ----------
    loglik : array-like
        Input data.
    d : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 7
    """
    loglik = np.atleast_1d(np.asarray(loglik, dtype=float))
    n = len(loglik)
    result = float(np.mean(loglik))
    se = float(np.std(loglik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BIC"})


def cheatsheet():
    return "eslbic: BIC"
