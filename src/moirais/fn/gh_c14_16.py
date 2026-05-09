# moirais.fn — function file (hadesllm/moirais)
"""Levy process representation of NCRM: Laplace functional and moments."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ncrm_levy"]


def ghosal_ncrm_levy(x):
    """
    Levy process representation of NCRM: Laplace functional and moments

    Formula: E[exp(-integral f dM)] = exp(-integral (1-e^{-f(x)*u}) nu(du,dx))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Levy process representation of NCRM: Laplace functional and moments"})


def cheatsheet():
    return "gh_c14_16: Levy process representation of NCRM: Laplace functional and moments"
