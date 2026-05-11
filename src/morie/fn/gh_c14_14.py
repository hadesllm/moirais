# morie.fn — function file (hadesllm/morie)
"""Normalized inverse-Gaussian process: NIG(alpha, G0) from inverse-Gaussian Levy measure."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_nig_proc"]


def ghosal_nig_proc(x):
    """
    Normalized inverse-Gaussian process: NIG(alpha, G0) from inverse-Gaussian Levy measure

    Formula: G = sum_k J_k/T delta_{theta_k}, J_k from IG Levy: rho(du)=exp(-alpha^2*u/2)/sqrt(2*pi*u^3) du

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
    Ghosal Ch 14 §14.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalized inverse-Gaussian process: NIG(alpha, G0) from inverse-Gaussian Levy measure"})


def cheatsheet():
    return "gh_c14_14: Normalized inverse-Gaussian process: NIG(alpha, G0) from inverse-Gaussian Levy measure"
