# morie.fn — function file (hadesllm/morie)
"""Prior via normalization of a completely random measure."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_norm_crm"]


def ghosal_norm_crm(x):
    """
    Prior via normalization of a completely random measure

    Formula: G(A) = M(A)/M(X), M ~ CRM with Levy measure nu

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
    Ghosal Ch 3 §3.4.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prior via normalization of a completely random measure"})


def cheatsheet():
    return "gh_c3_10: Prior via normalization of a completely random measure"
