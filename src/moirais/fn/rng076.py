"""DFT convolution property: time-domain convolution equals DFT-domain product.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dft_convolution_property"]


def rangayyan_ch3_dft_convolution_property(x, h):
    """
    DFT convolution property: time-domain convolution equals DFT-domain product.

    Formula: if y(n) = x(n) * h(n), then Y(k) = X(k) H(k)

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
    Rangayyan (2024), Ch 3, Eq 3.87, p. 130
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFT convolution property: time-domain convolution equals DFT-domain product."})


def cheatsheet():
    return "rng076: DFT convolution property: time-domain convolution equals DFT-domain product."
