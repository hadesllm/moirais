r"""Back translation prob.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_back_translation_prob"]


def kamath_ch3_back_translation_prob(t, thatt):
    r"""
    Back translation prob.

    Formula: P(t) = P_{forward}(\hat{t}|t) \cdot P_{backward}(t|\hat{t})

    Parameters
    ----------
    t : array-like
        Input data.
    thatt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.9, p. 105
    r"""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Back translation prob."})


def cheatsheet():
    return "km050: Back translation prob."
