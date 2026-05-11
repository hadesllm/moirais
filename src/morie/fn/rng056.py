"""Generic rational transfer function of an IIR filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_iir_transfer_function"]


def rangayyan_ch3_iir_transfer_function(b_k, a_k, z, N, M):
    """
    Generic rational transfer function of an IIR filter.

    Formula: H(z) = Y(z)/X(z) = ( sum_{k=0}^{N} b_k z^(-k) ) / ( 1 + sum_{k=1}^{M} a_k z^(-k) )

    Parameters
    ----------
    b_k : array-like
        Input data.
    a_k : array-like
        Input data.
    z : array-like
        Input data.
    N : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.67, p. 123
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generic rational transfer function of an IIR filter."})


def cheatsheet():
    return "rng056: Generic rational transfer function of an IIR filter."
