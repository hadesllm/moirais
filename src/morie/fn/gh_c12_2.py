# morie.fn — function file (hadesllm/morie)
"""BvM for Dirichlet process: Diaconis-Freedman approximation by Brownian bridge."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_bvm"]


def ghosal_dp_bvm(x):
    """
    BvM for Dirichlet process: Diaconis-Freedman approximation by Brownian bridge

    Formula: sqrt(n)(G_n(.) - F_0(.)) -> B(F_0(.)) weakly, B = standard Brownian bridge

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
    Ghosal Ch 12 §12.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BvM for Dirichlet process: Diaconis-Freedman approximation by Brownian bridge"})


def cheatsheet():
    return "gh_c12_2: BvM for Dirichlet process: Diaconis-Freedman approximation by Brownian bridge"
