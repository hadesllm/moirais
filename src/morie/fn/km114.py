"""Bleu precision.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_bleu_precision"]


def kamath_ch8_bleu_precision(n_grams):
    """
    Bleu precision.

    Formula: p_n = \frac{\text{# clipped matching n-grams}}{\text{# n-grams in generated text}}

    Parameters
    ----------
    n_grams : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.2, p. 323
    """
    n_grams = np.atleast_1d(np.asarray(n_grams, dtype=float))
    n = len(n_grams)
    result = float(np.mean(n_grams))
    se = float(np.std(n_grams, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bleu precision."})


def cheatsheet():
    return "km114: Bleu precision."
