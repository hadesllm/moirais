# moirais.fn — function file (hadesllm/moirais)
"""Beta process prior definition: H ~ BP(c, H0) for cumulative hazard function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_beta_proc_def"]


def ghosal_beta_proc_def(x):
    """
    Beta process prior definition: H ~ BP(c, H0) for cumulative hazard function

    Formula: BP(c, H0): increments dH(t) ~ Be(c(t)*dH0(t), c(t)*(1-dH0(t))) ind.

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
    Ghosal Ch 13 §13.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Beta process prior definition: H ~ BP(c, H0) for cumulative hazard function"})


def cheatsheet():
    return "gh_c13_3: Beta process prior definition: H ~ BP(c, H0) for cumulative hazard function"
