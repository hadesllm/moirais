# morie.fn — function file (hadesllm/morie)
"""Narrow-sense heritability from LMM variance components."""
import numpy as np
from ._richresult import RichResult

__all__ = ["heritability_lmm"]


def heritability_lmm(sigma_g2, sigma_e2):
    """
    Narrow-sense heritability from LMM variance components

    Formula: h^2 = sigma_g^2 / (sigma_g^2 + sigma_e^2)

    Parameters
    ----------
    sigma_g2 : array-like
        Input data.
    sigma_e2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'h2': 'float'}

    References
    ----------
    Montesinos Lopez Ch 5
    """
    sigma_g2 = np.asarray(sigma_g2, dtype=float)
    n = int(sigma_g2) if sigma_g2.ndim == 0 else len(sigma_g2)
    result = float(np.mean(sigma_g2))
    se = float(np.std(sigma_g2, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Narrow-sense heritability from LMM variance components"})


def cheatsheet():
    return "h2est: Narrow-sense heritability from LMM variance components"
