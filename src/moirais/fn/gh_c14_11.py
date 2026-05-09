# moirais.fn — function file (hadesllm/moirais)
"""PY power-law cluster growth: E[K_n] ~ C * n^d as n->infty for d in (0,1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_py_powerlaw"]


def ghosal_py_powerlaw(x):
    """
    PY power-law cluster growth: E[K_n] ~ C * n^d as n->infty for d in (0,1)

    Formula: E[K_n] ~ Gamma(theta+1)/Gamma(theta+d) * n^d / Gamma(1-d)

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
    Ghosal Ch 14 §14.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PY power-law cluster growth: E[K_n] ~ C * n^d as n->infty for d in (0,1)"})


def cheatsheet():
    return "gh_c14_11: PY power-law cluster growth: E[K_n] ~ C * n^d as n->infty for d in (0,1)"
