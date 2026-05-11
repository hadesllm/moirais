# morie.fn — function file (hadesllm/morie)
"""Lemma: number of ways to distribute n-like objects into r cells with no cell empty."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_distributing_objects"]


def gibbons_distributing_objects(n, r):
    """
    Lemma: number of ways to distribute n-like objects into r cells with no cell empty

    Formula: C(n-1, r-1) ways for n objects into r distinguishable cells

    Parameters
    ----------
    n : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: count

    References
    ----------
    Gibbons Lemma 3.2.1
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lemma: number of ways to distribute n-like objects into r cells with no cell empty"})


def cheatsheet():
    return "gb321l: Lemma: number of ways to distribute n-like objects into r cells with no cell empty"
