# morie.fn — function file (hadesllm/morie)
"""Pitman-Yor EPPF: prod (theta+j*d) / prod_{j=1}^k prod_{l=1}^{n_j-1}(l-d)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_py_eppf"]


def ghosal_py_eppf(x):
    """
    Pitman-Yor EPPF: prod (theta+j*d) / prod_{j=1}^k prod_{l=1}^{n_j-1}(l-d)

    Formula: p(n_1..n_k) = prod_{j=1}^{k-1}(theta+j*d) / prod_{i=1}^{n-1}(theta+i) * prod_{j=1}^k prod_{l=0}^{n_j-2}(l-d)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pitman-Yor EPPF: prod (theta+j*d) / prod_{j=1}^k prod_{l=1}^{n_j-1}(l-d)"})


def cheatsheet():
    return "gh_c14_10: Pitman-Yor EPPF: prod (theta+j*d) / prod_{j=1}^k prod_{l=1}^{n_j-1}(l-d)"
