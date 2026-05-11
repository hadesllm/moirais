"""Output alignment.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_output_alignment"]


def kamath_ch9_output_alignment(S_X):
    """
    Output alignment.

    Formula: H_X = OUT\_ALIGN_{T\to X}(S_X)

    Parameters
    ----------
    S_X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.19, p. 397
    """
    S_X = np.atleast_1d(np.asarray(S_X, dtype=float))
    n = len(S_X)
    result = float(np.mean(S_X))
    se = float(np.std(S_X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output alignment."})


def cheatsheet():
    return "km147: Output alignment."
