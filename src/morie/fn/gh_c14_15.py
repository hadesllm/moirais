# morie.fn -- function file (hadesllm/morie)
"""Normalized completely random measure: G = M(.)/M(X) for CRM M."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ncrm_def"]


def ghosal_ncrm_def(x):
    """
    Normalized completely random measure: G = M(.)/M(X) for CRM M

    Formula: G(A) = M(A)/M(X), M ~ CRM: M(A) = sum_{tau_k in A} J_k

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
    Ghosal Ch 14 §14.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalized completely random measure: G = M(.)/M(X) for CRM M"})


def cheatsheet():
    return "gh_c14_15: Normalized completely random measure: G = M(.)/M(X) for CRM M"
