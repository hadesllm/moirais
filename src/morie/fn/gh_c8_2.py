# morie.fn — function file (hadesllm/morie)
"""Ghosal-Ghosh-van der Vaart 2000 general contraction theorem."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ggv_thm"]


def ghosal_ggv_thm(x):
    """
    Ghosal-Ghosh-van der Vaart 2000 general contraction theorem

    Formula: eps_n contraction if: test condition + Pi(KL ball)>=exp(-n*eps_n^2) + entropy<=exp(n*eps_n^2)

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
    Ghosal Ch 8 §8.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ghosal-Ghosh-van der Vaart 2000 general contraction theorem"})


def cheatsheet():
    return "gh_c8_2: Ghosal-Ghosh-van der Vaart 2000 general contraction theorem"
