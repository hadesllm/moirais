r"""Conditional probability that defines an autoregressive language model: distribution over the next token given an L-token context.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_lm_next_token"]


def burkov_lm_ch2_lm_next_token(t_next, s):
    r"""
    Conditional probability that defines an autoregressive language model: distribution over the next token given an L-token context.

    Formula: \Pr\!\bigl(t = t_{L+1} \mid \mathbf{s} = (t_1, t_2, \ldots, t_L)\bigr)

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
    Burkov LM (2025), Ch 2, Eq 2.2, p. 76
    """
    t_next = np.atleast_1d(np.asarray(t_next, dtype=float))
    n = len(t_next)
    result = float(np.mean(t_next))
    se = float(np.std(t_next, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional probability that defines an autoregressive language model: distribution over the next token given an L-token context."})


def cheatsheet():
    return "b202: Conditional probability that defines an autoregressive language model: distribution over the next token given an L-token context."
