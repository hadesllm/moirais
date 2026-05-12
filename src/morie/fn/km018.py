r"""Layer norm.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_layer_norm"]


def kamath_ch2_layer_norm(h_i, mu, sigma, g):
    r"""
    Layer norm.

    Formula: h_i = g\,\frac{h_i - \mu}{\sigma}

    Parameters
    ----------
    h_i : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.18, p. 37
    r"""
    h_i = np.atleast_1d(np.asarray(h_i, dtype=float))
    n = len(h_i)
    result = float(np.mean(h_i))
    se = float(np.std(h_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Layer norm."})


def cheatsheet():
    return "km018: Layer norm."
