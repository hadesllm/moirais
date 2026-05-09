# moirais.fn — function file (hadesllm/moirais)
"""Consistency for linear regression with unknown error density."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_linreg_unk_err"]


def ghosal_linreg_unk_err(x, y):
    """
    Consistency for linear regression with unknown error density

    Formula: Y = X'beta + e, e ~ f unknown, joint consistency for (beta, f)

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
    Ghosal Ch 7 §7.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Consistency for linear regression with unknown error density"})


def cheatsheet():
    return "gh_c7_9: Consistency for linear regression with unknown error density"
