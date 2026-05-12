r"""Clm loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_clm_loss"]


def kamath_ch2_clm_loss(x):
    r"""
    Clm loss.

    Formula: L^{(x)}_{CLM} = -\frac{1}{|x|}\sum_{i=1}^{|x|} \log P(x_i|x_{<i})

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.21, p. 51
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clm loss."})


def cheatsheet():
    return "km021: Clm loss."
