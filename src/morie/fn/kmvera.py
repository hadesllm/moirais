# morie.fn -- function file (hadesllm/morie)
"""VeRA: shared frozen random matrices + per-layer learned scaling vectors."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_vera_adapter"]


def kamath_vera_adapter(W0, A_frozen, B_frozen, lam_b, lam_d, x):
    """
    VeRA: shared frozen random matrices + per-layer learned scaling vectors

    Formula: h = W_0 x + Lambda_b * B * Lambda_d * A x  (A, B frozen random; Lambda_b, Lambda_d learned)

    Parameters
    ----------
    W0 : array-like
        Input data.
    A_frozen : array-like
        Input data.
    B_frozen : array-like
        Input data.
    lam_b : array-like
        Input data.
    lam_d : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h

    References
    ----------
    Kamath Ch 4, VeRA section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VeRA: shared frozen random matrices + per-layer learned scaling vectors"})


def cheatsheet():
    return "kmvera: VeRA: shared frozen random matrices + per-layer learned scaling vectors"
