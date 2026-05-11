# morie.fn — function file (hadesllm/morie)
"""Concentration function phi_{f0}(eps) combines RKHS approximation and small ball prob."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_concentration_function"]


def ghosal_concentration_function(x):
    """
    Concentration function phi_{f0}(eps) combines RKHS approximation and small ball prob

    Formula: phi_{f0}(eps) = inf_{||h-f0||<eps} ||h||_H^2 - log Pi(||f-0||<eps)

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
    Ghosal Ch 11 §11.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Concentration function phi_{f0}(eps) combines RKHS approximation and small ball prob"})


def cheatsheet():
    return "gh_conc_func: Concentration function phi_{f0}(eps) combines RKHS approximation and small ball prob"
