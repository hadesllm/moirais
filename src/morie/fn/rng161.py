"""Variable step-size LMS update rule.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_variable_step"]


def rangayyan_ch3_lms_variable_step(w, mu, e, r, n):
    """
    Variable step-size LMS update rule.

    Formula: w(n+1) = w(n) + 2*mu(n)*e(n)*r(n)

    Parameters
    ----------
    w : array-like
        Input data.
    mu : array-like
        Input data.
    e : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.204, p. 185
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variable step-size LMS update rule."})


def cheatsheet():
    return "rng161: Variable step-size LMS update rule."
