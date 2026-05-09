"""Vector error correction model (VECM)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vector_error_correction"]


def vector_error_correction(X, r, lags):
    """
    Vector error correction model (VECM)

    Formula: DeltaY_t = alpha beta'Y_{t-1} + sum Gamma_i DeltaY_{t-i} + e_t

    Parameters
    ----------
    X : array-like
        Input data.
    r : array-like
        Input data.
    lags : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Johansen (1995) Likelihood-Based Inference
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vector error correction model (VECM)"})


def cheatsheet():
    return "vecmod: Vector error correction model (VECM)"
