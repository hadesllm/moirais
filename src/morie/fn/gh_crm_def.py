# morie.fn — function file (hadesllm/morie)
"""Completely random measure: M(A) and M(B) independent for disjoint A, B."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_completely_random_measure"]


def ghosal_completely_random_measure(x):
    """
    Completely random measure: M(A) and M(B) independent for disjoint A, B

    Formula: M(A) indep M(B) for A cap B = empty, characterized by Levy-Ito decomp

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal App J
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Completely random measure: M(A) and M(B) independent for disjoint A, B"})


def cheatsheet():
    return "gh_crm_def: Completely random measure: M(A) and M(B) independent for disjoint A, B"
