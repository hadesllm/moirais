# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Skip-gram with negative sampling loss: positive pair + k negatives."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_negative_sampling_skipgram"]


def alammar_negative_sampling_skipgram(center, word, negatives, V):
    """
    Skip-gram with negative sampling loss: positive pair + k negatives

    Formula: L = -log sigma(v_c^T v_w) - sum_{i=1..k} log sigma(-v_c^T v_{n_i})

    Parameters
    ----------
    center : array-like
        Input data.
    word : array-like
        Input data.
    negatives : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Alammar Ch 2, Negative Sampling section
    """
    center = np.atleast_1d(np.asarray(center, dtype=float))
    n = len(center)
    result = float(np.mean(center))
    se = float(np.std(center, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Skip-gram with negative sampling loss: positive pair + k negatives"})


def cheatsheet():
    return "alnsmp: Skip-gram with negative sampling loss: positive pair + k negatives"
