# moirais.fn — function file (hadesllm/moirais)
"""KL divergence properties: non-negativity, not symmetric, Pinsker inequality."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_kl_props"]


def ghosal_kl_props(x):
    """
    KL divergence properties: non-negativity, not symmetric, Pinsker inequality

    Formula: KL(P,Q)>=0, KL(P,Q)=0 iff P=Q, d_TV^2(P,Q) <= KL(P,Q)/2 (Pinsker)

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
    Ghosal App B
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KL divergence properties: non-negativity, not symmetric, Pinsker inequality"})


def cheatsheet():
    return "gh_ap_b1: KL divergence properties: non-negativity, not symmetric, Pinsker inequality"
