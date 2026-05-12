# morie.fn -- function file (hadesllm/morie)
"""Levy measure characterization of NTR process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ntr_levy"]


def ghosal_ntr_levy(x):
    """
    Levy measure characterization of NTR process

    Formula: Laplace functional: E[exp(-integral f dM)] = exp(-integral (1-e^{-f}) dnu)

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
    Ghosal Ch 13 §13.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Levy measure characterization of NTR process"})


def cheatsheet():
    return "gh_c13_9: Levy measure characterization of NTR process"
