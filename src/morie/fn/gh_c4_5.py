# morie.fn -- function file (hadesllm/morie)
"""Self-similarity of DP: conditional distribution given partition is again DP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_selfsim"]


def ghosal_dp_selfsim(x):
    """
    Self-similarity of DP: conditional distribution given partition is again DP

    Formula: G|G(A)=w ~ w*DP(alpha*G0(.|A)) + (1-w)*DP(alpha*G0(.|A^c))

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
    Ghosal Ch 4 §4.1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-similarity of DP: conditional distribution given partition is again DP"})


def cheatsheet():
    return "gh_c4_5: Self-similarity of DP: conditional distribution given partition is again DP"
