# moirais.fn — function file (hadesllm/moirais)
"""Countable Dirichlet process on simplex: normalization of gamma random variables."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dir_simplex"]


def ghosal_dir_simplex(x):
    """
    Countable Dirichlet process on simplex: normalization of gamma random variables

    Formula: (p_1..p_k) ~ Dir(alpha_1..alpha_k): p_j = G_j/sum G_i, G_j ~ Ga(alpha_j,1)

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
    Ghosal Ch 3 §3.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Countable Dirichlet process on simplex: normalization of gamma random variables"})


def cheatsheet():
    return "gh_c3_3: Countable Dirichlet process on simplex: normalization of gamma random variables"
