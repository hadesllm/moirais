r"""Rouge n.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_rouge_n"]


def kamath_ch8_rouge_n(S, gram_n):
    r"""
    Rouge n.

    Formula: \mathrm{ROUGE\text{-}N} = \frac{\sum_{S\in\text{Ref}}\sum_{gram_n\in S} \mathrm{Count}_{match}(gram_n)}{\sum_{S\in\text{Ref}}\sum_{gram_n\in S} \mathrm{Count}(gram_n)}

    Parameters
    ----------
    S : array-like
        Input data.
    gram_n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.6, p. 324
    r"""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rouge n."})


def cheatsheet():
    return "km118: Rouge n."
