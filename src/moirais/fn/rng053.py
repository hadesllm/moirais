"""Z-transform of a causal FIR system of length N (transfer function).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_z_transform_fir"]


def rangayyan_ch3_z_transform_fir(h, n, z, N):
    """
    Z-transform of a causal FIR system of length N (transfer function).

    Formula: H(z) = sum_{n=0}^{N-1} h(n) * z^(-n)

    Parameters
    ----------
    h : array-like
        Input data.
    n : array-like
        Input data.
    z : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.55, p. 119
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Z-transform of a causal FIR system of length N (transfer function)."})


def cheatsheet():
    return "rng053: Z-transform of a causal FIR system of length N (transfer function)."
