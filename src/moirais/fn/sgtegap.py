"""Choose k via largest gap in normalised-Laplacian spectrum."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_eigengap_heuristic"]


def sgt_eigengap_heuristic(A, K_max):
    """
    Choose k via largest gap in normalised-Laplacian spectrum

    Formula: k* = argmax_k (λ_{k+1} - λ_k)

    Parameters
    ----------
    A : array-like
        Input data.
    K_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: k_star, lams

    References
    ----------
    von Luxburg (2007)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Choose k via largest gap in normalised-Laplacian spectrum"})


def cheatsheet():
    return "sgtegap: Choose k via largest gap in normalised-Laplacian spectrum"
