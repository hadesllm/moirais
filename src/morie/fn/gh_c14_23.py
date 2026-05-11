# morie.fn — function file (hadesllm/morie)
"""Indian buffet process: feature allocation model for latent binary matrices."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ibp_def"]


def ghosal_ibp_def(x):
    """
    Indian buffet process: feature allocation model for latent binary matrices

    Formula: Z in {0,1}^{n x K}: P(feature k assigned to customer i) from IBP

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
    Ghosal Ch 14 §14.10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Indian buffet process: feature allocation model for latent binary matrices"})


def cheatsheet():
    return "gh_c14_23: Indian buffet process: feature allocation model for latent binary matrices"
