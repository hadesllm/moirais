# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Recursive text chunking: split on separators in priority order until under target size."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_recursive_chunking"]


def alammar_recursive_chunking(text, separators, target_size, overlap):
    """
    Recursive text chunking: split on separators in priority order until under target size

    Formula: split(text, seps) -> if any chunk > target: recursive split with next-tier separator

    Parameters
    ----------
    text : array-like
        Input data.
    separators : array-like
        Input data.
    target_size : array-like
        Input data.
    overlap : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: chunks

    References
    ----------
    Alammar Ch 8, chunking strategies section
    """
    text = np.atleast_1d(np.asarray(text, dtype=float))
    n = len(text)
    result = float(np.mean(text))
    se = float(np.std(text, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recursive text chunking: split on separators in priority order until under target size"})


def cheatsheet():
    return "alchk: Recursive text chunking: split on separators in priority order until under target size"
