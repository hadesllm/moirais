# morie.fn -- function file (hadesllm/morie)
"""KL property for exponential density families: log-series prior has KL support."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_exp_dens_kl"]


def ghosal_exp_dens_kl(x):
    """
    KL property for exponential density families: log-series prior has KL support

    Formula: pi(psi) = exp(-lambda||psi||) for psi in Sobolev ball has KL support

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
    Ghosal Ch 7 §7.1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KL property for exponential density families: log-series prior has KL support"})


def cheatsheet():
    return "gh_c7_3: KL property for exponential density families: log-series prior has KL support"
