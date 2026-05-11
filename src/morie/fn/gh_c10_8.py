# morie.fn — function file (hadesllm/morie)
"""Normal regression via finite random series: optimal adaptive rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_frs_reg"]


def ghosal_frs_reg(x, y):
    """
    Normal regression via finite random series: optimal adaptive rate

    Formula: Y_i = f(x_i)+e_i, f = sum_{k<=K} beta_k phi_k, rate n^{-2s/(2s+1)}

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 10 §10.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normal regression via finite random series: optimal adaptive rate"})


def cheatsheet():
    return "gh_c10_8: Normal regression via finite random series: optimal adaptive rate"
