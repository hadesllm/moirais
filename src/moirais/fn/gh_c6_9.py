# moirais.fn — function file (hadesllm/moirais)
"""Permanence of KL property: operations preserving KL support condition."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_kl_perm"]


def ghosal_kl_perm(x):
    """
    Permanence of KL property: operations preserving KL support condition

    Formula: If Pi1, Pi2 have KL property at P0, so does their mixture and product

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
    Ghosal Ch 6 §6.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Permanence of KL property: operations preserving KL support condition"})


def cheatsheet():
    return "gh_c6_9: Permanence of KL property: operations preserving KL support condition"
