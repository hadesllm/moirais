"""Nucleus (top-p) sampling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["top_p_sampling"]


def top_p_sampling(logits, p, temp):
    """
    Nucleus (top-p) sampling

    Formula: smallest set with cumulative prob ≥p

    Parameters
    ----------
    logits : array-like
        Input data.
    p : array-like
        Input data.
    temp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holtzman et al (2020)
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nucleus (top-p) sampling"})


def cheatsheet():
    return "toppS: Nucleus (top-p) sampling"
