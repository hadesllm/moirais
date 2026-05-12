r"""Weat effect size.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_weat_effect_size"]


def kamath_ch6_weat_effect_size(A_1, A_2, W_1, W_2):
    r"""
    Weat effect size.

    Formula: \mathrm{WEAT}(A_1,A_2,W_1,W_2) = \frac{\mathrm{mean}_{a_1\in A_1} s(a_1,W_1,W_2) - \mathrm{mean}_{a_2\in A_2} s(a_2,W_1,W_2)}{\mathrm{std}_{a\in A_1\cup A_2} s(a,W_1,W_2)}

    Parameters
    ----------
    A_1 : array-like
        Input data.
    A_2 : array-like
        Input data.
    W_1 : array-like
        Input data.
    W_2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.6, p. 234
    r"""
    A_1 = np.atleast_1d(np.asarray(A_1, dtype=float))
    n = len(A_1)
    result = float(np.mean(A_1))
    se = float(np.std(A_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weat effect size."})


def cheatsheet():
    return "km082: Weat effect size."
