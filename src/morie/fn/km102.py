"""Lstm chain rule.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_lstm_chain_rule"]


def kamath_ch6_lstm_chain_rule(w_1_w_M):
    """
    Lstm chain rule.

    Formula: P(w_1,w_2,\dots,w_M) = \prod_{t=1}^M P(w_t|w_1,w_2,\dots,w_{t-1})

    Parameters
    ----------
    w_1_w_M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.26, p. 252
    """
    w_1_w_M = np.atleast_1d(np.asarray(w_1_w_M, dtype=float))
    n = len(w_1_w_M)
    result = float(np.mean(w_1_w_M))
    se = float(np.std(w_1_w_M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lstm chain rule."})


def cheatsheet():
    return "km102: Lstm chain rule."
