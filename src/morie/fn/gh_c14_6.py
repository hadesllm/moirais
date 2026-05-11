# morie.fn — function file (hadesllm/morie)
"""SSP posterior distribution: conjugate-like update for species sampling process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ssp_post"]


def ghosal_ssp_post(x):
    """
    SSP posterior distribution: conjugate-like update for species sampling process

    Formula: G|X_1..X_n from SSP with updated weights for observed species

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
    Ghosal Ch 14 §14.2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SSP posterior distribution: conjugate-like update for species sampling process"})


def cheatsheet():
    return "gh_c14_6: SSP posterior distribution: conjugate-like update for species sampling process"
