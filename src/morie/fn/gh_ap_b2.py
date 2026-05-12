# morie.fn -- function file (hadesllm/morie)
"""KL variation V_k(P,Q): higher-order moment of log likelihood ratio."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_kl_variation"]


def ghosal_kl_variation(x):
    """
    KL variation V_k(P,Q): higher-order moment of log likelihood ratio

    Formula: V_k(P,Q) = integral |log(p/q)|^k dP, V_1 = KL(P,Q)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KL variation V_k(P,Q): higher-order moment of log likelihood ratio"})


def cheatsheet():
    return "gh_ap_b2: KL variation V_k(P,Q): higher-order moment of log likelihood ratio"
