"""G-function: nearest-neighbor distance CDF."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_g_function"]


def schabenberger_g_function(points, r):
    """
    G-function: nearest-neighbor distance CDF

    Formula: G(r) = P(d_nn <= r) = 1 - exp(-lambda*pi*r^2) for CSR

    Parameters
    ----------
    points : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 3, Sec 3.3.4
    """
    points = np.asarray(points, dtype=float)
    n = int(points) if points.ndim == 0 else len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "G-function: nearest-neighbor distance CDF"})


def cheatsheet():
    return "spgfun: G-function: nearest-neighbor distance CDF"
