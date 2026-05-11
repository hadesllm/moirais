"""Cloze prompt template.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_cloze_prompt_template"]


def kamath_ch3_cloze_prompt_template(x, z):
    """
    Cloze prompt template.

    Formula: x' = [x] \text{ This is a } [z] \text{ movie.}

    Parameters
    ----------
    x : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.7, p. 101
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cloze prompt template."})


def cheatsheet():
    return "km048: Cloze prompt template."
