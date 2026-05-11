"""Numbered display equation (8.1) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_1"]


def mvsml_categorical_count_eq_8_1(f, H, the, square, of, norm):
    """
    Numbered display equation (8.1) from MVSML chapter 8.

    Formula: f H is the square of the norm of f(xi) on H, a measure of model complexity (de los Campos et al. 2010). Hilbert spaces are complete linear spaces endowed with a norm that is the square root of the inner product in the space. The Hilbert spaces that are relevant for our discussion are RKHS of real-valued functions, here denoted as H. Those interested in more technical details of RKHS of real functions should read Wahba (1990). By the representer theorem (Wahba 1990), which tells us that the solutions to some regularization functionals in high or inﬁnite-dimensional spaces fall in a ﬁnite- dimensional space, the solution for

    Parameters
    ----------
    f : array-like
        Input data.
    H : array-like
        Input data.
    the : array-like
        Input data.
    square : array-like
        Input data.
    of : array-like
        Input data.
    norm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.1) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.1) from MVSML chapter 8."})


def cheatsheet():
    return "msm124: Numbered display equation (8.1) from MVSML chapter 8."
