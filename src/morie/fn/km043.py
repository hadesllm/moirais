r"""Prompt softmax label.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_prompt_softmax_label"]


def kamath_ch3_prompt_softmax_label(w, h_z, M):
    r"""
    Prompt softmax label.

    Formula: p(y|x) = \frac{\exp(w_{M(y)}\cdot h_z)}{\sum_{y'\in y}\exp(w_{M(y')}\cdot h_z)}

    Parameters
    ----------
    w : array-like
        Input data.
    h_z : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.2, p. 91
    r"""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prompt softmax label."})


def cheatsheet():
    return "km043: Prompt softmax label."
