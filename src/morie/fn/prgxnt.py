"""Perplexity of language model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["perplexity"]


def perplexity(log_probs, N):
    """
    Perplexity of language model

    Formula: PPL = exp(-1/N sum log p(x_i))

    Parameters
    ----------
    log_probs : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Brown et al (1992)
    """
    log_probs = np.atleast_1d(np.asarray(log_probs, dtype=float))
    n = len(log_probs)
    result = float(np.mean(log_probs))
    se = float(np.std(log_probs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perplexity of language model"})


def cheatsheet():
    return "prgxnt: Perplexity of language model"
