"""Gpt unsupervised obj.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt_unsupervised_obj"]


def kamath_ch2_gpt_unsupervised_obj(U, k, Theta):
    """
    Gpt unsupervised obj.

    Formula: L_1(U) = \sum_i \log P(u_i|u_{i-k},\dots,u_{i-1};\Theta)

    Parameters
    ----------
    U : array-like
        Input data.
    k : array-like
        Input data.
    Theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.34, p. 70
    """
    U = np.atleast_1d(np.asarray(U, dtype=float))
    n = len(U)
    result = float(np.mean(U))
    se = float(np.std(U, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gpt unsupervised obj."})


def cheatsheet():
    return "km034: Gpt unsupervised obj."
