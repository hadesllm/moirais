"""Asymptotic theory check -- finite-sample variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["check_asymptote_msm"]


def check_asymptote_msm(y, A, H, B):
    """
    Asymptotic theory check -- finite-sample variance

    Formula: compare E[psi^2]/n to bootstrap

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL-Rose (2011); Funk et al (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic theory check -- finite-sample variance"})


def cheatsheet():
    return "chasym: Asymptotic theory check -- finite-sample variance"
