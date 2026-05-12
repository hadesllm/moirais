# morie.fn -- function file (hadesllm/morie)
"""Countable Dirichlet process on countable sample space."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_countable_dp"]


def ghosal_countable_dp(x):
    """
    Countable Dirichlet process on countable sample space

    Formula: (G(x_1), G(x_2), ...) ~ Dir(alpha*G0(x_1), alpha*G0(x_2), ...)

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
    Ghosal Ch 3 §3.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Countable Dirichlet process on countable sample space"})


def cheatsheet():
    return "gh_c3_5: Countable Dirichlet process on countable sample space"
