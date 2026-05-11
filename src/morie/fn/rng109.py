"""Transfer function of the recursive 8-point MA filter (sinc-like).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_8point_recursive_transfer_function"]


def rangayyan_ch3_ma_8point_recursive_transfer_function(z):
    """
    Transfer function of the recursive 8-point MA filter (sinc-like).

    Formula: H(z) = (1/8) * (1 - z^(-8)) / (1 - z^(-1))

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.121, p. 145
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transfer function of the recursive 8-point MA filter (sinc-like)."})


def cheatsheet():
    return "rng109: Transfer function of the recursive 8-point MA filter (sinc-like)."
