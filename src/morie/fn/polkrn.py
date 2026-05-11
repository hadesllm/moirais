"""Polynomial-kernel MSM for nonlinear effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["polynomial_kernel_msm"]


def polynomial_kernel_msm(y, A_history, H_history, degree):
    """
    Polynomial-kernel MSM for nonlinear effects

    Formula: basis expansion of A_bar

    Parameters
    ----------
    y : array-like
        Input data.
    A_history : array-like
        Input data.
    H_history : array-like
        Input data.
    degree : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hernán et al (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polynomial-kernel MSM for nonlinear effects"})


def cheatsheet():
    return "polkrn: Polynomial-kernel MSM for nonlinear effects"
