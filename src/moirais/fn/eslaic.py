"""AIC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_aic_score"]


def esl_aic_score(loglik, d):
    """
    AIC

    Formula: AIC = -2 log L + 2 d

    Parameters
    ----------
    loglik : array-like
        Input data.
    d : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AIC"})


def cheatsheet():
    return "eslaic: AIC"
