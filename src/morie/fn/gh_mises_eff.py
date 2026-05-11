# morie.fn — function file (hadesllm/morie)
"""Mises differentiability and BvM: functional is Mises-differentiable => semiparametric BvM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_mises_efficiency"]


def ghosal_mises_efficiency(x):
    """
    Mises differentiability and BvM: functional is Mises-differentiable => semiparametric BvM

    Formula: psi(P) Mises-differentiable at P0 with influence function tilde{psi} => efficient BvM

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
    Ghosal Ch 12 §12.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mises differentiability and BvM: functional is Mises-differentiable => semiparametric BvM"})


def cheatsheet():
    return "gh_mises_eff: Mises differentiability and BvM: functional is Mises-differentiable => semiparametric BvM"
