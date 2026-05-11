# morie.fn — function file (hadesllm/morie)
"""Batch size heuristic: power of two in [32, 512] balancing noise and throughput."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_batch_size_heuristic"]


def geron_batch_size_heuristic(n_train):
    """
    Batch size heuristic: power of two in [32, 512] balancing noise and throughput

    Formula: B in {32, 64, 128, 256, 512}

    Parameters
    ----------
    n_train : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: batch_size

    References
    ----------
    Géron Ch 9
    """
    n_train = np.atleast_1d(np.asarray(n_train, dtype=float))
    n = len(n_train)
    result = float(np.mean(n_train))
    se = float(np.std(n_train, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Batch size heuristic: power of two in [32, 512] balancing noise and throughput"})


def cheatsheet():
    return "hmbsz: Batch size heuristic: power of two in [32, 512] balancing noise and throughput"
