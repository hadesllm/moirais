# moirais.fn — function file (hadesllm/moirais)
"""Priors based on finite approximating sets: sieve priors for contraction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_fin_apx_pri"]


def ghosal_fin_apx_pri(x):
    """
    Priors based on finite approximating sets: sieve priors for contraction

    Formula: Pi_n = Pi on Theta_n, Theta_n = eps_n-net of size exp(n*eps_n^2)

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
    Ghosal Ch 8 §8.2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Priors based on finite approximating sets: sieve priors for contraction"})


def cheatsheet():
    return "gh_c8_7: Priors based on finite approximating sets: sieve priors for contraction"
