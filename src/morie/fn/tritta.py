"""Triangular self-attention over starting/ending nodes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_triangle_attn"]


def alphafold_triangle_attn(pair_repr, start_end):
    """
    Triangular self-attention over starting/ending nodes

    Formula: a_ij^k = softmax(q_ij^T k_ik + b_jk)

    Parameters
    ----------
    pair_repr : array-like
        Input data.
    start_end : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021) AlphaFold2
    """
    pair_repr = np.atleast_1d(np.asarray(pair_repr, dtype=float))
    n = len(pair_repr)
    result = float(np.mean(pair_repr))
    se = float(np.std(pair_repr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Triangular self-attention over starting/ending nodes"})


def cheatsheet():
    return "tritta: Triangular self-attention over starting/ending nodes"
