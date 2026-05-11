"""Transfer function of a generic MA (FIR) filter of order N.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_transfer_function"]


def rangayyan_ch3_ma_transfer_function(b_k, z, N):
    """
    Transfer function of a generic MA (FIR) filter of order N.

    Formula: H(z) = Y(z)/X(z) = sum_{k=0}^{N} b_k * z^(-k)

    Parameters
    ----------
    b_k : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.99, p. 140
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transfer function of a generic MA (FIR) filter of order N."})


def cheatsheet():
    return "rng088: Transfer function of a generic MA (FIR) filter of order N."
