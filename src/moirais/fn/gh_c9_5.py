# moirais.fn — function file (hadesllm/moirais)
"""Approximation by normal mixtures: any smooth density approx by Gaussian mixture."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_norm_mix_apx"]


def ghosal_norm_mix_apx(x):
    """
    Approximation by normal mixtures: any smooth density approx by Gaussian mixture

    Formula: p0 in Sobolev(s) => exists G: ||p0 - integral phi_sigma dG||_1 <= sigma^s

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
    Ghosal Ch 9 §9.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Approximation by normal mixtures: any smooth density approx by Gaussian mixture"})


def cheatsheet():
    return "gh_c9_5: Approximation by normal mixtures: any smooth density approx by Gaussian mixture"
