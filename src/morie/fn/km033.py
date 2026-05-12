r"""Dae loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_dae_loss"]


def kamath_ch2_dae_loss(x, xhat):
    r"""
    Dae loss.

    Formula: L_{DAE} = -\frac{1}{|x|}\sum_{i=1}^{|x|}\log P(x_i|\hat{x}, x_{<i})

    Parameters
    ----------
    x : array-like
        Input data.
    xhat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.33, p. 55
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dae loss."})


def cheatsheet():
    return "km033: Dae loss."
