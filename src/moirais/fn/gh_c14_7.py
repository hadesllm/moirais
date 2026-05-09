# moirais.fn — function file (hadesllm/moirais)
"""SSP mixture models: density f(x) = integral K(x;theta) dG(theta), G ~ SSP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ssp_mix"]


def ghosal_ssp_mix(x):
    """
    SSP mixture models: density f(x) = integral K(x;theta) dG(theta), G ~ SSP

    Formula: f(x) = integral K(x;theta) dG(theta), G = sum p_k delta_{theta_k} ~ SSP

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
    Ghosal Ch 14 §14.2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SSP mixture models: density f(x) = integral K(x;theta) dG(theta), G ~ SSP"})


def cheatsheet():
    return "gh_c14_7: SSP mixture models: density f(x) = integral K(x;theta) dG(theta), G ~ SSP"
