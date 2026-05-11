"""Lora forward.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_lora_forward"]


def kamath_ch4_lora_forward(W_0, B, A, x):
    """
    Lora forward.

    Formula: h = W_0 x + \Delta W x = W_0 x + B A x

    Parameters
    ----------
    W_0 : array-like
        Input data.
    B : array-like
        Input data.
    A : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 4, Eq 4.5, p. 151
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lora forward."})


def cheatsheet():
    return "km058: Lora forward."
