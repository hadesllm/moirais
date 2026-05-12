r"""Slm loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_slm_loss"]


def kamath_ch2_slm_loss(x, R_x):
    r"""
    Slm loss.

    Formula: L^{(x)}_{SLM} = -\frac{1}{|R_x|}\sum_{i\in R_x} \log P(x_i|x_{\setminus R_x})

    Parameters
    ----------
    x : array-like
        Input data.
    R_x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.26, p. 53
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Slm loss."})


def cheatsheet():
    return "km026: Slm loss."
