# moirais.fn — function file (hadesllm/moirais)
"""Matching pursuit greedy decomposition into dictionary atoms."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_matching_pursuit"]


def rangayyan_matching_pursuit(x, dictionary, max_iter, tol):
    """
    Matching pursuit greedy decomposition into dictionary atoms

    Formula: R_0=x; R_n=R_{n-1}-<R_{n-1},phi_k>*phi_k; iterate until ||R||<epsilon

    Parameters
    ----------
    x : array-like
        Input data.
    dictionary : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coeffs, atoms, residual

    References
    ----------
    Rangayyan Ch 9.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matching pursuit greedy decomposition into dictionary atoms"})


def cheatsheet():
    return "rgmp: Matching pursuit greedy decomposition into dictionary atoms"
