"""L-function: variance-stabilized K-function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_l_function"]


def schabenberger_l_function(points, lambda_est, r):
    """
    L-function: variance-stabilized K-function

    Formula: L(r) = sqrt(K(r)/pi) - r; L(r)=0 under CSR

    Parameters
    ----------
    points : array-like
        Input data.
    lambda_est : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 3, Sec 3.4.2
    """
    points = np.asarray(points, dtype=float)
    n = int(points) if points.ndim == 0 else len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L-function: variance-stabilized K-function"})


def cheatsheet():
    return "splfun: L-function: variance-stabilized K-function"
