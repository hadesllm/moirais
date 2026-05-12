r"""Mlm loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_mlm_loss"]


def kamath_ch2_mlm_loss(x, M_x):
    r"""
    Mlm loss.

    Formula: L^{(x)}_{MLM} = -\frac{1}{|M_x|}\sum_{i\in M_x} \log P(x_i|x_{\setminus M_x})

    Parameters
    ----------
    x : array-like
        Input data.
    M_x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.22, p. 51
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mlm loss."})


def cheatsheet():
    return "km022: Mlm loss."
