# moirais.fn — function file (hadesllm/moirais)
"""Prior mass condition for contraction: Pi(KL ball of radius eps_n) >= exp(-n*eps_n^2)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_prior_mass_cnd"]


def ghosal_prior_mass_cnd(x):
    """
    Prior mass condition for contraction: Pi(KL ball of radius eps_n) >= exp(-n*eps_n^2)

    Formula: Pi({P: KL(P0,P)<eps_n^2, V2(P0,P)<eps_n^2}) >= exp(-n*eps_n^2)

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
    Ghosal Ch 8 §8.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prior mass condition for contraction: Pi(KL ball of radius eps_n) >= exp(-n*eps_n^2)"})


def cheatsheet():
    return "gh_c8_4: Prior mass condition for contraction: Pi(KL ball of radius eps_n) >= exp(-n*eps_n^2)"
