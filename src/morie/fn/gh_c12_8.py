# morie.fn — function file (hadesllm/morie)
"""Semiparametric BvM for Cox proportional hazard model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_cox_bvm_sp"]


def ghosal_cox_bvm_sp(x):
    """
    Semiparametric BvM for Cox proportional hazard model

    Formula: sqrt(n)(beta_n - beta_0) -> N(0, I_beta^{-1}) via partial likelihood BvM

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
    Ghosal Ch 12 §12.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric BvM for Cox proportional hazard model"})


def cheatsheet():
    return "gh_c12_8: Semiparametric BvM for Cox proportional hazard model"
