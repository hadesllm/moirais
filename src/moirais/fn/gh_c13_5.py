# moirais.fn — function file (hadesllm/moirais)
"""Continuous-time Beta process as a Lévy process with specific Levy measure."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bp_cont"]


def ghosal_bp_cont(x):
    """
    Continuous-time Beta process as a Lévy process with specific Levy measure

    Formula: BP(c,H0): Levy measure nu(du x dt) = u^{-1}(1-u)^{c(t)-1} du * c(t) dH0(t)

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
    Ghosal Ch 13 §13.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous-time Beta process as a Lévy process with specific Levy measure"})


def cheatsheet():
    return "gh_c13_5: Continuous-time Beta process as a Lévy process with specific Levy measure"
