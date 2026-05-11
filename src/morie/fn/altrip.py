# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""SBERT triplet loss: anchor-positive closer than anchor-negative by margin."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_sbert_triplet_loss"]


def alammar_sbert_triplet_loss(anchor, positive, negative, margin):
    """
    SBERT triplet loss: anchor-positive closer than anchor-negative by margin

    Formula: L = max( 0, d(a, p) - d(a, n) + margin )

    Parameters
    ----------
    anchor : array-like
        Input data.
    positive : array-like
        Input data.
    negative : array-like
        Input data.
    margin : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Alammar Ch 10, Triplet Loss section
    """
    anchor = np.atleast_1d(np.asarray(anchor, dtype=float))
    n = len(anchor)
    result = float(np.mean(anchor))
    se = float(np.std(anchor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SBERT triplet loss: anchor-positive closer than anchor-negative by margin"})


def cheatsheet():
    return "altrip: SBERT triplet loss: anchor-positive closer than anchor-negative by margin"
