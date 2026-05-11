"""Unidirectional encoder state.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_unidirectional_encoder_state"]


def kamath_ch2_unidirectional_encoder_state(h_t_1, x_t):
    """
    Unidirectional encoder state.

    Formula: h_t = f(h_{t-1}, x_t)

    Parameters
    ----------
    h_t_1 : array-like
        Input data.
    x_t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.1, p. 30
    """
    h_t_1 = np.atleast_1d(np.asarray(h_t_1, dtype=float))
    n = len(h_t_1)
    result = float(np.mean(h_t_1))
    se = float(np.std(h_t_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unidirectional encoder state."})


def cheatsheet():
    return "km001: Unidirectional encoder state."
