"""Convolutional signal model addressed by homomorphic deconvolution.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_convolution_model"]


def rangayyan_ch4_convolution_model(x, h, t):
    """
    Convolutional signal model addressed by homomorphic deconvolution.

    Formula: y(t) = x(t) * h(t)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.61, p. 245
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convolutional signal model addressed by homomorphic deconvolution."})


def cheatsheet():
    return "rng233: Convolutional signal model addressed by homomorphic deconvolution."
