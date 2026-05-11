"""Input projector.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_input_projector"]


def kamath_ch9_input_projector(F_X):
    """
    Input projector.

    Formula: P_X = IN\_ALIGN_{X\to T}(F_X)

    Parameters
    ----------
    F_X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.3, p. 380
    """
    F_X = np.atleast_1d(np.asarray(F_X, dtype=float))
    n = len(F_X)
    result = float(np.mean(F_X))
    se = float(np.std(F_X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Input projector."})


def cheatsheet():
    return "km131: Input projector."
