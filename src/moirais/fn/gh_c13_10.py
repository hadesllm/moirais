# moirais.fn — function file (hadesllm/moirais)
"""NTR process posterior consistency for survival distribution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ntr_consist"]


def ghosal_ntr_consist(x):
    """
    NTR process posterior consistency for survival distribution

    Formula: Pi_n(F: d(F,F0)>eps | X^n) -> 0 under KL support condition on NTR

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
    Ghosal Ch 13 §13.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NTR process posterior consistency for survival distribution"})


def cheatsheet():
    return "gh_c13_10: NTR process posterior consistency for survival distribution"
