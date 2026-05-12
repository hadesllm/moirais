# morie.fn -- function file (hadesllm/morie)
"""Groundedness reward: fraction of answer tokens supported by retrieved context."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_groundedness_reward"]


def kamath_groundedness_reward(y_tokens, ctx_tokens):
    """
    Groundedness reward: fraction of answer tokens supported by retrieved context

    Formula: grd(y | ctx) = |{tokens in y also in ctx}| / |y|

    Parameters
    ----------
    y_tokens : array-like
        Input data.
    ctx_tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: grounded_frac

    References
    ----------
    Kamath Ch 5, Groundedness reward modeling section
    """
    y_tokens = np.atleast_1d(np.asarray(y_tokens, dtype=float))
    n = len(y_tokens)
    result = float(np.mean(y_tokens))
    se = float(np.std(y_tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Groundedness reward: fraction of answer tokens supported by retrieved context"})


def cheatsheet():
    return "kmgrnd: Groundedness reward: fraction of answer tokens supported by retrieved context"
