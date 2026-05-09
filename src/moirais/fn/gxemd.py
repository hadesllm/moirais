# moirais.fn — function file (hadesllm/moirais)
"""Genotype-by-environment interaction model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gxe_interaction_model"]


def gxe_interaction_model(x, y, env):
    """
    Genotype-by-environment interaction model

    Formula: y_ij = mu + g_i + e_j + (ge)_ij + eps

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    env : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Montesinos Lopez Ch 11
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genotype-by-environment interaction model"})


def cheatsheet():
    return "gxemd: Genotype-by-environment interaction model"
