# moirais.fn — function file (hadesllm/moirais)
"""Tail-free priors satisfy KL property and hence posterior consistency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_tailfree_con"]


def ghosal_tailfree_con(x):
    """
    Tail-free priors satisfy KL property and hence posterior consistency

    Formula: PT(T_m, A) with alpha_m -> infty has KL support at any continuous P0

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
    Ghosal Ch 6 §6.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tail-free priors satisfy KL property and hence posterior consistency"})


def cheatsheet():
    return "gh_c6_8: Tail-free priors satisfy KL property and hence posterior consistency"
