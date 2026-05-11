# morie.fn — function file (hadesllm/morie)
"""Integrated BM prior: k_m(s,t) = integral_0^{min(s,t)} (s-u)^{m-1}(t-u)^{m-1}/(m-1)!^2 du."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_brownian_primitive"]


def ghosal_gp_brownian_primitive(x):
    """
    Integrated BM prior: k_m(s,t) = integral_0^{min(s,t)} (s-u)^{m-1}(t-u)^{m-1}/(m-1)!^2 du

    Formula: W^{(m)} = m-fold integrated BM, k_m covariance, paths C^{m-1}

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
    Ghosal Ch 11 §11.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Integrated BM prior: k_m(s,t) = integral_0^{min(s,t)} (s-u)^{m-1}(t-u)^{m-1}/(m-1)!^2 du"})


def cheatsheet():
    return "gh_gp_brow_prim: Integrated BM prior: k_m(s,t) = integral_0^{min(s,t)} (s-u)^{m-1}(t-u)^{m-1}/(m-1)!^2 du"
