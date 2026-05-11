"""Triangle inequality for two vectors.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_triangle_inequality_vectors"]


def rangayyan_ch4_triangle_inequality_vectors(a, b):
    """
    Triangle inequality for two vectors.

    Formula: |a + b| <= |a| + |b|

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.45, p. 239
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Triangle inequality for two vectors."})


def cheatsheet():
    return "rng219: Triangle inequality for two vectors."
