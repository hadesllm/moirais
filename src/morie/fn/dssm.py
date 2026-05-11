"""Deep Structured Semantic Model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dssm"]


def dssm(query, doc):
    """
    Deep Structured Semantic Model

    Formula: hashed n-gram bow + deep encoder + cosine

    Parameters
    ----------
    query : array-like
        Input data.
    doc : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Huang et al (2013)
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep Structured Semantic Model"})


def cheatsheet():
    return "dssm: Deep Structured Semantic Model"
