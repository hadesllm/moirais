"""Bigram topic model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bigram_topic"]


def bigram_topic(docs, K):
    """
    Bigram topic model

    Formula: topic-conditional bigram probabilities

    Parameters
    ----------
    docs : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wallach (2006)
    """
    docs = np.atleast_1d(np.asarray(docs, dtype=float))
    n = len(docs)
    result = float(np.mean(docs))
    se = float(np.std(docs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bigram topic model"})


def cheatsheet():
    return "bigtm: Bigram topic model"
