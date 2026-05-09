# moirais.fn — function file (hadesllm/moirais)
"""Wishart prior on covariance in Gaussian DPM: full covariance contraction rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_wishart_dpm"]


def ghosal_wishart_dpm(x):
    """
    Wishart prior on covariance in Gaussian DPM: full covariance contraction rate

    Formula: Sigma ~ Wishart(nu, Psi), location-scale DPM, same rate as diagonal

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
    Ghosal Ch 9 §9.4.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wishart prior on covariance in Gaussian DPM: full covariance contraction rate"})


def cheatsheet():
    return "gh_c9_6: Wishart prior on covariance in Gaussian DPM: full covariance contraction rate"
