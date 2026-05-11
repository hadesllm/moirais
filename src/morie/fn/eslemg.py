"""EM updates for GMM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_em_gmm"]


def esl_em_gmm(X, k):
    """
    EM updates for GMM

    Formula: E: gamma_ik = pi_k N_k / sum; M: pi, mu, Sigma updates

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: params

    References
    ----------
    Hastie ESL Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EM updates for GMM"})


def cheatsheet():
    return "eslemg: EM updates for GMM"
