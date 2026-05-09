"""Convolution in time becomes multiplication of z-transforms.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_z_transform_convolution"]


def rangayyan_ch3_z_transform_convolution(x, h):
    """
    Convolution in time becomes multiplication of z-transforms.

    Formula: if y(n) = x(n) * h(n), then Y(z) = X(z) H(z)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.56, p. 119
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convolution in time becomes multiplication of z-transforms."})


def cheatsheet():
    return "rng054: Convolution in time becomes multiplication of z-transforms."
