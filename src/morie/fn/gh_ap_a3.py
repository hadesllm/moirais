# morie.fn — function file (hadesllm/morie)
"""Total variation distance: dTV(P,Q) = sup_A |P(A)-Q(A)| = 1/2 * integral |p-q|."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_tv_distance"]


def ghosal_tv_distance(x):
    """
    Total variation distance: dTV(P,Q) = sup_A |P(A)-Q(A)| = 1/2 * integral |p-q|

    Formula: d_TV(P,Q) = sup_{A measurable} |P(A)-Q(A)| = (1/2)||p-q||_1

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
    Ghosal App A
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Total variation distance: dTV(P,Q) = sup_A |P(A)-Q(A)| = 1/2 * integral |p-q|"})


def cheatsheet():
    return "gh_ap_a3: Total variation distance: dTV(P,Q) = sup_A |P(A)-Q(A)| = 1/2 * integral |p-q|"
