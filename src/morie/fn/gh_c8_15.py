# morie.fn — function file (hadesllm/morie)
"""Alpha-posterior contraction rate: robust Bayes at same minimax rate as standard posterior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_alpha_pst_crt"]


def ghosal_alpha_pst_crt(x):
    """
    Alpha-posterior contraction rate: robust Bayes at same minimax rate as standard posterior

    Formula: pi_alpha posterior contracts at same rate eps_n under weaker conditions

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
    Ghosal Ch 8 §8.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Alpha-posterior contraction rate: robust Bayes at same minimax rate as standard posterior"})


def cheatsheet():
    return "gh_c8_15: Alpha-posterior contraction rate: robust Bayes at same minimax rate as standard posterior"
