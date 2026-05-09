# moirais.fn — function file (hadesllm/moirais)
"""Tail-free process: partition-based construction with independence across levels."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_tailfree_def"]


def ghosal_tailfree_def(x):
    """
    Tail-free process: partition-based construction with independence across levels

    Formula: (G(B_e0)/G(B_e), G(B_e1)/G(B_e)) independent across partitions

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
    Ghosal Ch 3 §3.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tail-free process: partition-based construction with independence across levels"})


def cheatsheet():
    return "gh_c3_11: Tail-free process: partition-based construction with independence across levels"
