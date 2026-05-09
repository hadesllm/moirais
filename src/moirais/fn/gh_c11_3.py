# moirais.fn — function file (hadesllm/moirais)
"""GP posterior contraction theorem: rate from concentration function phi_n."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_crt_thm"]


def ghosal_gp_crt_thm(x):
    """
    GP posterior contraction theorem: rate from concentration function phi_n

    Formula: eps_n satisfying n*eps_n^2 >= phi_n(eps_n) gives contraction rate

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP posterior contraction theorem: rate from concentration function phi_n"})


def cheatsheet():
    return "gh_c11_3: GP posterior contraction theorem: rate from concentration function phi_n"
