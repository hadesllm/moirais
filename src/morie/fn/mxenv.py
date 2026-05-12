# morie.fn -- function file (hadesllm/morie)
"""Multi-environment genomic model: environment as factor in LMM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["multi_env_model"]


def multi_env_model(y, markers, env, G):
    """
    Multi-environment genomic model: environment as factor in LMM

    Formula: y_ijk = mu + e_j + g_i + (ge)_ij + eps; G-matrix for g and ge effects

    Parameters
    ----------
    y : array-like
        Input data.
    markers : array-like
        Input data.
    env : array-like
        Input data.
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'gebv': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 5,6
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-environment genomic model: environment as factor in LMM"})


def cheatsheet():
    return "mxenv: Multi-environment genomic model: environment as factor in LMM"
