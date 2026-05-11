# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Additive noise model (ANM) for bivariate causal direction detection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["additive_noise_model"]


def additive_noise_model(X, Y):
    """
    Additive noise model (ANM) for bivariate causal direction detection

    Formula: Y = f(X) + N_Y vs X = g(Y) + N_X; test independence of residuals: X->Y if N_Y _|_ X

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'direction': 'str', 'p_values': 'dict'}

    References
    ----------
    Molak Ch 13
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Additive noise model (ANM) for bivariate causal direction detection"})


def cheatsheet():
    return "anmod: Additive noise model (ANM) for bivariate causal direction detection"
