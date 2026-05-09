"""Dante cloze.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_dante_cloze"]


def kamath_ch3_dante_cloze(prompt):
    """
    Dante cloze.

    Formula: \text{Dante was born in [MASK]}

    Parameters
    ----------
    prompt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.4, p. 95
    """
    prompt = np.atleast_1d(np.asarray(prompt, dtype=float))
    n = len(prompt)
    result = float(np.mean(prompt))
    se = float(np.std(prompt, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dante cloze."})


def cheatsheet():
    return "km045: Dante cloze."
