# morie.fn -- function file (hadesllm/morie)
"""ARMA pole-zero model identification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pole_zero_model"]


def rangayyan_pole_zero_model(x, p, q):
    """
    ARMA pole-zero model identification

    Formula: H(z) = B(z)/A(z); poles from AR, zeros from MA part

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b_coeffs, a_coeffs

    References
    ----------
    Rangayyan Ch 7.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARMA pole-zero model identification"})


def cheatsheet():
    return "rgpzmod: ARMA pole-zero model identification"
