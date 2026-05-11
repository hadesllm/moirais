"""Gender direction.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_gender_direction"]


def kamath_ch6_gender_direction(A, E):
    """
    Gender direction.

    Formula: g = \frac{1}{|A|}\sum_{(a_i,a_j)\in A} (E(a_j) - E(a_i))

    Parameters
    ----------
    A : array-like
        Input data.
    E : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.19, p. 243
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gender direction."})


def cheatsheet():
    return "km095: Gender direction."
