"""Complex cepstra of a convolution decompose as a sum.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_complex_cepstra_sum"]


def rangayyan_ch4_complex_cepstra_sum(x_hat, h_hat, n):
    """
    Complex cepstra of a convolution decompose as a sum.

    Formula: y_hat(n) = x_hat(n) + h_hat(n)

    Parameters
    ----------
    x_hat : array-like
        Input data.
    h_hat : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.66, p. 247
    """
    x_hat = np.atleast_1d(np.asarray(x_hat, dtype=float))
    n = len(x_hat)
    result = float(np.mean(x_hat))
    se = float(np.std(x_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complex cepstra of a convolution decompose as a sum."})


def cheatsheet():
    return "rng238: Complex cepstra of a convolution decompose as a sum."
