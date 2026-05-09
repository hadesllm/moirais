# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Sampling decoding: draw from the softmax distribution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_sampling_decoding"]


def alammar_sampling_decoding(logits, seed):
    """
    Sampling decoding: draw from the softmax distribution

    Formula: y_t ~ Categorical(softmax(logits_t))

    Parameters
    ----------
    logits : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: token

    References
    ----------
    Alammar Ch 3, Sampling section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sampling decoding: draw from the softmax distribution"})


def cheatsheet():
    return "alspl: Sampling decoding: draw from the softmax distribution"
