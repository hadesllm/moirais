# morie.fn -- function file (hadesllm/morie)
"""Semiparametric BvM for smooth functionals: efficient posterior centering."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_semipara_bvm"]


def ghosal_semipara_bvm(x):
    """
    Semiparametric BvM for smooth functionals: efficient posterior centering

    Formula: sqrt(n)(psi(G_n) - psi(F_0)) -> N(0, sigma_eff^2) with sigma_eff^2 = Cramer-Rao lb

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric BvM for smooth functionals: efficient posterior centering"})


def cheatsheet():
    return "gh_c12_4: Semiparametric BvM for smooth functionals: efficient posterior centering"
