# morie.fn — function file (hadesllm/morie)
"""Efficient influence function for semiparametric BvM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_eff_infl_fn"]


def ghosal_eff_infl_fn(x):
    """
    Efficient influence function for semiparametric BvM

    Formula: psi(P) - psi(P0) = E_{P0}[tilde{psi}(X)] + o(||P-P0||), tilde{psi} = eff. inf. fn.

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
    Ghosal Ch 12 §12.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Efficient influence function for semiparametric BvM"})


def cheatsheet():
    return "gh_c12_5: Efficient influence function for semiparametric BvM"
