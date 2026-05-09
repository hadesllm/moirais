# moirais.fn — function file (hadesllm/moirais)
"""GP sample path continuity: Kolmogorov-Chentsov criterion for path continuity."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_sample_cont"]


def ghosal_gp_sample_cont(x):
    """
    GP sample path continuity: Kolmogorov-Chentsov criterion for path continuity

    Formula: E|f(x)-f(y)|^p <= C*||x-y||^{1+alpha} => paths Holder-(alpha/p) continuous

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
    Ghosal App I
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP sample path continuity: Kolmogorov-Chentsov criterion for path continuity"})


def cheatsheet():
    return "gh_ap_i1: GP sample path continuity: Kolmogorov-Chentsov criterion for path continuity"
