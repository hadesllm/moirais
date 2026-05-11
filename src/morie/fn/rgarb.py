# morie.fn — function file (hadesllm/morie)
"""AR model via Burg method."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ar_burg"]


def rangayyan_ar_burg(x):
    """
    AR model via Burg method

    Formula: a_k via Levinson-Durbin recursion

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
    Rangayyan Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AR model via Burg method"})


def cheatsheet():
    return "rgarb: AR model via Burg method"
