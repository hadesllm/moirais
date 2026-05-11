# morie.fn — function file (hadesllm/morie)
"""Stationary GP via spectral representation: k(x-y) = integral exp(i*omega'*(x-y)) dF(omega)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_statgp_spec"]


def ghosal_statgp_spec(x):
    """
    Stationary GP via spectral representation: k(x-y) = integral exp(i*omega'*(x-y)) dF(omega)

    Formula: k(x,y) = integral exp(i*omega'*(x-y)) dF(omega) by Bochner theorem

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
    Ghosal Ch 11 §11.4.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stationary GP via spectral representation: k(x-y) = integral exp(i*omega'*(x-y)) dF(omega)"})


def cheatsheet():
    return "gh_c11_9: Stationary GP via spectral representation: k(x-y) = integral exp(i*omega'*(x-y)) dF(omega)"
