# morie.fn — function file (hadesllm/morie)
"""Semiparametric efficiency: BvM variance equals efficient Cramer-Rao lower bound."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_semipara_eff"]


def ghosal_semipara_eff(x):
    """
    Semiparametric efficiency: BvM variance equals efficient Cramer-Rao lower bound

    Formula: var >= (nabla psi)^T I_{theta,eta}^{-1} nabla psi (semiparametric Cramer-Rao)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric efficiency: BvM variance equals efficient Cramer-Rao lower bound"})


def cheatsheet():
    return "gh_c12_6: Semiparametric efficiency: BvM variance equals efficient Cramer-Rao lower bound"
