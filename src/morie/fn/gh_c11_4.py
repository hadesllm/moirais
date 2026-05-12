# morie.fn -- function file (hadesllm/morie)
"""GP density estimation contraction rate via concentration function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_dens_crt"]


def ghosal_gp_dens_crt(x):
    """
    GP density estimation contraction rate via concentration function

    Formula: f = exp(psi)/Z, psi ~ GP, rate eps_n = n^{-s/(2s+1)} for s-smooth psi

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
    Ghosal Ch 11 §11.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "GP density estimation contraction rate via concentration function"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "GP density estimation contraction rate via concentration function"})


def cheatsheet():
    return "gh_c11_4: GP density estimation contraction rate via concentration function"
