# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Greedy decoding: argmax at each step."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_greedy_decoding"]


def alammar_greedy_decoding(logits):
    """
    Greedy decoding: argmax at each step

    Formula: y_t = argmax_v p_theta(v | y_{<t}, x)

    Parameters
    ----------
    logits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: token

    References
    ----------
    Alammar Ch 3, Greedy decoding section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Greedy decoding: argmax at each step"})


def cheatsheet():
    return "algrdy: Greedy decoding: argmax at each step"
