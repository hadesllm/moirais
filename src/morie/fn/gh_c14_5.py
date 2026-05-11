# morie.fn — function file (hadesllm/morie)
"""Species sampling process: G = sum_k p_k delta_{theta_k} with random weights (p_k)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ssp_def"]


def ghosal_ssp_def(x):
    """
    Species sampling process: G = sum_k p_k delta_{theta_k} with random weights (p_k)

    Formula: G = sum_{k=1}^infty p_k delta_{theta_k}, theta_k iid G0, sum p_k=1 a.s.

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
    Ghosal Ch 14 §14.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Species sampling process: G = sum_k p_k delta_{theta_k} with random weights (p_k)"})


def cheatsheet():
    return "gh_c14_5: Species sampling process: G = sum_k p_k delta_{theta_k} with random weights (p_k)"
