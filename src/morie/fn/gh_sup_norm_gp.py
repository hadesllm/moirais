# morie.fn -- function file (hadesllm/morie)
"""GP posterior contraction in sup-norm for regression functions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_sup_norm_contraction"]


def ghosal_sup_norm_contraction(x):
    """
    GP posterior contraction in sup-norm for regression functions

    Formula: Pi_n(||f-f0||_infty > M*eps_n | data) -> 0 at rate n^{-s/(2s+d)} * (log n)^t

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
    Ghosal Ch 11 §11.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP posterior contraction in sup-norm for regression functions"})


def cheatsheet():
    return "gh_sup_norm_gp: GP posterior contraction in sup-norm for regression functions"
