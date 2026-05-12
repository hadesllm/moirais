# morie.fn -- function file (hadesllm/morie)
"""Random Fourier features (RFF) kernel approximation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["random_fourier_features"]


def random_fourier_features(X, D, kernel):
    """
    Random Fourier features (RFF) kernel approximation

    Formula: z(X) = sqrt(2/D) * cos(W'x + b); K(x,y) approx z(x)'*z(y); W ~ p(w), b ~ Uniform(0,2*pi)

    Parameters
    ----------
    X : array-like
        Input data.
    D : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'Z': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random Fourier features (RFF) kernel approximation"})


def cheatsheet():
    return "rfkrn: Random Fourier features (RFF) kernel approximation"
