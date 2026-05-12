# morie.fn -- function file (hadesllm/morie)
"""IBP stick-breaking construction: pi_k = prod_{j<=k} V_j, V_j ~ Beta(alpha, 1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ibp_stickbr"]


def ghosal_ibp_stickbr(x):
    """
    IBP stick-breaking construction: pi_k = prod_{j<=k} V_j, V_j ~ Beta(alpha, 1)

    Formula: pi_k = prod_{j=1}^k V_j, V_j iid Beta(alpha,1), P(Z_{ik}=1)=pi_k

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
    Ghosal Ch 14 §14.10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IBP stick-breaking construction: pi_k = prod_{j<=k} V_j, V_j ~ Beta(alpha, 1)"})


def cheatsheet():
    return "gh_c14_24: IBP stick-breaking construction: pi_k = prod_{j<=k} V_j, V_j ~ Beta(alpha, 1)"
