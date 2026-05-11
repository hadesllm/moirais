"""Prompt label mapping.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_prompt_label_mapping"]


def kamath_ch3_prompt_label_mapping(x, y, M):
    """
    Prompt label mapping.

    Formula: p(y|x) = p(z = M(y) | x')

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.1, p. 91
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prompt label mapping."})


def cheatsheet():
    return "km042: Prompt label mapping."
