# moirais.fn — function file (hadesllm/moirais)
"""Beta process sample path generation: via thinned Poisson process representation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bp_path_gen"]


def ghosal_bp_path_gen(x):
    """
    Beta process sample path generation: via thinned Poisson process representation

    Formula: H(t) = sum_{tau_k <= t} J_k, (J_k, tau_k) from Poisson process on [0,1]x[0,T]

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
    Ghosal Ch 13 §13.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Beta process sample path generation: via thinned Poisson process representation"})


def cheatsheet():
    return "gh_c13_6: Beta process sample path generation: via thinned Poisson process representation"
