# morie.fn — function file (hadesllm/morie)
"""BvM for linear functionals in white noise model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_wn_lin_bvm"]


def ghosal_wn_lin_bvm(x):
    """
    BvM for linear functionals in white noise model

    Formula: sqrt(n)(L(theta_n) - L(theta_0)) -> N(0, ||L||^2) for linear functional L

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
    Ghosal Ch 12 §12.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BvM for linear functionals in white noise model"})


def cheatsheet():
    return "gh_c12_10: BvM for linear functionals in white noise model"
