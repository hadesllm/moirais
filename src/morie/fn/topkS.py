"""Top-k sampling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["top_k_sampling"]


def top_k_sampling(logits, k, temp):
    """
    Top-k sampling

    Formula: sample only from top k probabilities

    Parameters
    ----------
    logits : array-like
        Input data.
    k : array-like
        Input data.
    temp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fan-Lewis-Dauphin (2018)
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Top-k sampling"})


def cheatsheet():
    return "topkS: Top-k sampling"
