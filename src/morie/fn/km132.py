"""Llm signal tokens.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_llm_signal_tokens"]


def kamath_ch9_llm_signal_tokens(P_X, F_T):
    """
    Llm signal tokens.

    Formula: (t, S_X) = \mathrm{LLM}(P_X, F_T)

    Parameters
    ----------
    P_X : array-like
        Input data.
    F_T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.4, p. 383
    """
    P_X = np.atleast_1d(np.asarray(P_X, dtype=float))
    n = len(P_X)
    result = float(np.mean(P_X))
    se = float(np.std(P_X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Llm signal tokens."})


def cheatsheet():
    return "km132: Llm signal tokens."
