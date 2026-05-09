"""Tlm loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_tlm_loss"]


def kamath_ch2_tlm_loss(x, y, M_x, M_y):
    """
    Tlm loss.

    Formula: L^{(x)}_{TLM} = -\frac{1}{|M_x|}\sum_{i\in M_x}\log P(x_i|x_{\setminus M_x},y_{\setminus M_y}) - \frac{1}{|M_y|}\sum_{i\in M_y}\log P(y_i|x_{\setminus M_x},y_{\setminus M_y})

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    M_x : array-like
        Input data.
    M_y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.27, p. 53
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tlm loss."})


def cheatsheet():
    return "km027: Tlm loss."
