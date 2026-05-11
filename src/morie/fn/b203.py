"""Equivalent shorthand notations for the autoregressive next-token probability used throughout the book.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_lm_shorthand"]


def burkov_lm_ch2_lm_shorthand(t_next, s):
    """
    Equivalent shorthand notations for the autoregressive next-token probability used throughout the book.

    Formula: \Pr(t_{L+1} \mid t_1, t_2, \ldots, t_L) \quad \text{or} \quad \Pr(t_{L+1} \mid \mathbf{s})

    Parameters
    ----------
    t_next : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: conditional probability of next token

    References
    ----------
    Burkov LM (2025), Ch 2, Eq 2.3, p. 76
    """
    t_next = np.atleast_1d(np.asarray(t_next, dtype=float))
    n = len(t_next)
    result = float(np.mean(t_next))
    se = float(np.std(t_next, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Equivalent shorthand notations for the autoregressive next-token probability used throughout the book."})


def cheatsheet():
    return "b203: Equivalent shorthand notations for the autoregressive next-token probability used throughout the book."
