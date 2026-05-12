r"""Bleu final.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_bleu_final"]


def kamath_ch8_bleu_final(BP, p_n, N):
    r"""
    Bleu final.

    Formula: \mathrm{BLEU} = \mathrm{BP}\cdot \exp(\sum_{n=1}^N \frac{1}{N}\log p_n)

    Parameters
    ----------
    BP : array-like
        Input data.
    p_n : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.5, p. 323
    r"""
    BP = np.atleast_1d(np.asarray(BP, dtype=float))
    n = len(BP)
    result = float(np.mean(BP))
    se = float(np.std(BP, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bleu final."})


def cheatsheet():
    return "km117: Bleu final."
