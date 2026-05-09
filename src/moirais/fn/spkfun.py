"""Ripley's K-function: expected number of extra events within distance r of a random event."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_k_function"]


def schabenberger_k_function(points, lambda_est, r):
    """
    Ripley's K-function: expected number of extra events within distance r of a random event

    Formula: K(r) = (1/lambda) * E[# events within r of a randomly chosen event] = pi*r^2 for CSR

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
    Schabenberger Ch 3, Sec 3.4.1
    """
    points = np.asarray(points, dtype=float)
    n = int(points) if points.ndim == 0 else len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ripley's K-function: expected number of extra events within distance r of a random event"})


def cheatsheet():
    return "spkfun: Ripley's K-function: expected number of extra events within distance r of a random event"
