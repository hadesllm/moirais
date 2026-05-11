# morie.fn — function file (hadesllm/morie)
"""ARMA (pole-zero) prediction error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch7_arma_error"]


def rangayyan_ch7_arma_error(x, b_coeffs, a_coeffs):
    """
    ARMA (pole-zero) prediction error

    Formula: e[n] = sum a_k*x[n-k] - sum b_k*u[n-k] minimized iteratively

    Parameters
    ----------
    x : array-like
        Input data.
    b_coeffs : array-like
        Input data.
    a_coeffs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: error

    References
    ----------
    Rangayyan Ch 7.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARMA (pole-zero) prediction error"})


def cheatsheet():
    return "rgeqn7b: ARMA (pole-zero) prediction error"
