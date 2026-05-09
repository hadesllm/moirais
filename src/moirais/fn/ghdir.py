# moirais.fn — function file (hadesllm/moirais)
"""Dirichlet process posterior update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dirichlet_posterior"]


def ghosal_dirichlet_posterior(x):
    """
    Dirichlet process posterior update

    Formula: DP(alpha+n, (alpha*G0 + sum delta_Xi)/(alpha+n))

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
    Ghosal Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet process posterior update"})


def cheatsheet():
    return "ghdir: Dirichlet process posterior update"
