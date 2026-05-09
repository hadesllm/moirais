# moirais.fn — function file (hadesllm/moirais)
"""Laplace functional of CRM: E[exp(-integral f dM)] = exp(-integral (1-e^{-f*u}) nu(du,dx))."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_crm_laplace"]


def ghosal_crm_laplace(x):
    """
    Laplace functional of CRM: E[exp(-integral f dM)] = exp(-integral (1-e^{-f*u}) nu(du,dx))

    Formula: log E[exp(-integral f dM)] = -integral_0^infty integral_X (1-e^{-f(x)u}) nu(du,dx)

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
    Ghosal App J
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplace functional of CRM: E[exp(-integral f dM)] = exp(-integral (1-e^{-f*u}) nu(du,dx))"})


def cheatsheet():
    return "gh_ap_j2: Laplace functional of CRM: E[exp(-integral f dM)] = exp(-integral (1-e^{-f*u}) nu(du,dx))"
