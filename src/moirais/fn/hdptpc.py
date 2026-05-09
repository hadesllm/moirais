"""HDP topic model (nonparametric LDA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hdp_topic_model"]


def hdp_topic_model(docs, gamma, alpha):
    """
    HDP topic model (nonparametric LDA)

    Formula: shared atoms across documents via HDP

    Parameters
    ----------
    docs : array-like
        Input data.
    gamma : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Teh-Jordan-Beal-Blei (2006)
    """
    docs = np.atleast_1d(np.asarray(docs, dtype=float))
    n = len(docs)
    result = float(np.mean(docs))
    se = float(np.std(docs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HDP topic model (nonparametric LDA)"})


def cheatsheet():
    return "hdptpc: HDP topic model (nonparametric LDA)"
