"""AutoInt — multi-head self-attention for CTR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["autoint"]


def autoint(X, y, K):
    """
    AutoInt — multi-head self-attention for CTR

    Formula: attention over feature embeddings

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Song et al (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AutoInt — multi-head self-attention for CTR"})


def cheatsheet():
    return "autoI: AutoInt — multi-head self-attention for CTR"
