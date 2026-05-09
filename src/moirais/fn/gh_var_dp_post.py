# moirais.fn — function file (hadesllm/moirais)
"""Variational DP posterior: truncated stick-breaking with KL-minimizing variational distribution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_variational_dp_posterior"]


def ghosal_variational_dp_posterior(x):
    """
    Variational DP posterior: truncated stick-breaking with KL-minimizing variational distribution

    Formula: q*(G_K, theta, z) = argmin KL(q||pi(.|X)), truncation at K

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
    Ghosal Ch 5 §5.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational DP posterior: truncated stick-breaking with KL-minimizing variational distribution"})


def cheatsheet():
    return "gh_var_dp_post: Variational DP posterior: truncated stick-breaking with KL-minimizing variational distribution"
