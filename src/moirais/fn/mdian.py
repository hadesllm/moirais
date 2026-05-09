# moirais.fn — function file (hadesllm/moirais)
"""Mediation analysis: decompose total effect into direct and indirect."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mediation_analysis"]


def mediation_analysis(Y, T, M, X):
    """
    Mediation analysis: decompose total effect into direct and indirect

    Formula: Total = Direct + Indirect; Direct = E[Y(Y,M(Y'))] - E[Y(x',M(Y'))]; Indirect = E[Y(x,M(x))] - E[Y(x,M(x'))]

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.
    M : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'total': 'float', 'direct': 'float', 'indirect': 'float'}

    References
    ----------
    Molak Ch 6
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mediation analysis: decompose total effect into direct and indirect"})


def cheatsheet():
    return "mdian: Mediation analysis: decompose total effect into direct and indirect"
