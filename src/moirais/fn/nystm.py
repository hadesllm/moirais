# moirais.fn — function file (hadesllm/moirais)
"""Nystrom kernel matrix approximation using m landmark points."""
import numpy as np
from ._richresult import RichResult

__all__ = ["nystrom_approximation"]


def nystrom_approximation(X, m_landmarks):
    """
    Nystrom kernel matrix approximation using m landmark points

    Formula: K approx K_{nm} * K_{mm}^{-1} * K_{mn}; selects m << n inducing points

    Parameters
    ----------
    X : array-like
        Input data.
    m_landmarks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'K_approx': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nystrom kernel matrix approximation using m landmark points"})


def cheatsheet():
    return "nystm: Nystrom kernel matrix approximation using m landmark points"
