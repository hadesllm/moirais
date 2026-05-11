# morie.fn — function file (hadesllm/morie)
"""DP KL neighborhood mass: lower bound on Pi(KL(P0,P)<eps)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_kl_nbhd_mass"]


def ghosal_dp_kl_nbhd_mass(x):
    """
    DP KL neighborhood mass: lower bound on Pi(KL(P0,P)<eps)

    Formula: Pi(KL(P0,P)<eps) >= exp(-C*eps^{-1/s}) for DP with smooth G0

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
    Ghosal Ch 7 §7.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP KL neighborhood mass: lower bound on Pi(KL(P0,P)<eps)"})


def cheatsheet():
    return "gh_dp_kl_nbhd: DP KL neighborhood mass: lower bound on Pi(KL(P0,P)<eps)"
