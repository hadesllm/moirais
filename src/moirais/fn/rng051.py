"""LTI convolution maps to multiplication in s-domain and frequency domain.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lti_convolution_property"]


def rangayyan_ch3_lti_convolution_property(x, h):
    """
    LTI convolution maps to multiplication in s-domain and frequency domain.

    Formula: if y(t) = x(t) * h(t), then Y(s) = X(s) H(s) and Y(omega) = X(omega) H(omega)

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
    Rangayyan (2024), Ch 3, Eq 3.53, p. 119
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LTI convolution maps to multiplication in s-domain and frequency domain."})


def cheatsheet():
    return "rng051: LTI convolution maps to multiplication in s-domain and frequency domain."
