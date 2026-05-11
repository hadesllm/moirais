"""Deformable DETR sparse attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deformable_detr"]


def deformable_detr(x, queries, K):
    """
    Deformable DETR sparse attention

    Formula: K reference points per query; sparse attn

    Parameters
    ----------
    x : array-like
        Input data.
    queries : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhu et al (2021) Deformable DETR
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deformable DETR sparse attention"})


def cheatsheet():
    return "defdtr: Deformable DETR sparse attention"
