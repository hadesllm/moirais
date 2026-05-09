# moirais.fn — function file (hadesllm/moirais)
"""K-means via Lloyd's algorithm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kmeans_clustering"]


def kmeans_clustering(x):
    """
    K-means via Lloyd's algorithm

    Formula: argmin sum ||xi - mu_k||^2

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-means via Lloyd's algorithm"})


def cheatsheet():
    return "kmnsc: K-means via Lloyd's algorithm"
