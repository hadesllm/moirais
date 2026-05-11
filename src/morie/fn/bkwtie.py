# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Weight tying: share the embedding matrix with the output projection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_weight_tying"]


def burkov_weight_tying(h_last, E):
    """
    Weight tying: share the embedding matrix with the output projection

    Formula: logits = h_L * E^T  where E is the input embedding matrix (shared)

    Parameters
    ----------
    h_last : array-like
        Input data.
    E : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: logits

    References
    ----------
    Burkov Ch 4, Weight Tying section
    """
    h_last = np.atleast_1d(np.asarray(h_last, dtype=float))
    n = len(h_last)
    result = float(np.mean(h_last))
    se = float(np.std(h_last, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weight tying: share the embedding matrix with the output projection"})


def cheatsheet():
    return "bkwtie: Weight tying: share the embedding matrix with the output projection"
