# morie.fn -- function file (hadesllm/morie)
"""Mixtures of Beta processes for flexible hazard modeling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_mix_bp"]


def ghosal_mix_bp(x):
    """
    Mixtures of Beta processes for flexible hazard modeling

    Formula: H ~ integral BP(c, H0_lambda) dPi(lambda)

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
    Ghosal Ch 13 §13.3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixtures of Beta processes for flexible hazard modeling"})


def cheatsheet():
    return "gh_c13_7: Mixtures of Beta processes for flexible hazard modeling"
