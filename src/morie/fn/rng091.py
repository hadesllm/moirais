"""Z-domain expression for the Hann filter output.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_hann_z_output"]


def rangayyan_ch3_hann_z_output(X, z):
    """
    Z-domain expression for the Hann filter output.

    Formula: Y(z) = (1/4) * [X(z) + 2*z^(-1)*X(z) + z^(-2)*X(z)]

    Parameters
    ----------
    X : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.102, p. 140
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Z-domain expression for the Hann filter output."})


def cheatsheet():
    return "rng091: Z-domain expression for the Hann filter output."
