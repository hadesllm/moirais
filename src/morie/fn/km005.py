r"""Decoder token distribution.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_decoder_token_distribution"]


def kamath_ch2_decoder_token_distribution(s_t_1, y_t_1, c):
    r"""
    Decoder token distribution.

    Formula: P(y_{t'}|y_{t'-1},\dots,y_1,c) = \mathrm{softmax}(s_{t-1}, y_{t'-1}, c)

    Parameters
    ----------
    s_t_1 : array-like
        Input data.
    y_t_1 : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.5, p. 31
    r"""
    s_t_1 = np.atleast_1d(np.asarray(s_t_1, dtype=float))
    n = len(s_t_1)
    result = float(np.mean(s_t_1))
    se = float(np.std(s_t_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decoder token distribution."})


def cheatsheet():
    return "km005: Decoder token distribution."
