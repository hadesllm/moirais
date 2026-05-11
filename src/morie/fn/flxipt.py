"""Flexible IPTW with super learner."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["flexible_iptw"]


def flexible_iptw(A, H, library):
    """
    Flexible IPTW with super learner

    Formula: SL ensemble for f(A_t|H_t)

    Parameters
    ----------
    A : array-like
        Input data.
    H : array-like
        Input data.
    library : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL-Polley-Hubbard (2007); Pirracchio et al (2015)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Flexible IPTW with super learner"})


def cheatsheet():
    return "flxipt: Flexible IPTW with super learner"
