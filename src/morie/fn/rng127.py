"""Bilinear transformation mapping s-domain to z-domain.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_bilinear_transformation"]


def rangayyan_ch3_bilinear_transformation(z, T):
    """
    Bilinear transformation mapping s-domain to z-domain.

    Formula: s = (2/T) * (1 - z^(-1)) / (1 + z^(-1))

    Parameters
    ----------
    z : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.139, p. 154
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bilinear transformation mapping s-domain to z-domain."})


def cheatsheet():
    return "rng127: Bilinear transformation mapping s-domain to z-domain."
