# moirais.fn — function file (hadesllm/moirais)
"""Perplexity of a model on a token sequence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_perplexity"]


def kamath_perplexity(log_probs):
    """
    Perplexity of a model on a token sequence

    Formula: PPL = exp( - (1/N) sum_{t=1..N} log p_theta(x_t | x_{<t}) )

    Parameters
    ----------
    log_probs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ppl

    References
    ----------
    Kamath Ch 8, Perplexity section
    """
    log_probs = np.atleast_1d(np.asarray(log_probs, dtype=float))
    n = len(log_probs)
    result = float(np.mean(log_probs))
    se = float(np.std(log_probs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perplexity of a model on a token sequence"})


def cheatsheet():
    return "kmperp: Perplexity of a model on a token sequence"
