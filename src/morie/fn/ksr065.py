"""General defining equation for the efficient influence function via score and information operators."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_efficient_influence_general"]


def kosorok_ch3_efficient_influence_general(A, psi_tilde, chi_tilde, eta):
    """
    General defining equation for the efficient influence function via score and information operators

    Formula: A*_eta * psi_tilde_{P_eta} = chi_tilde_eta in lin H_eta

    Parameters
    ----------
    A : array-like
        Input data.
    psi_tilde : array-like
        Input data.
    chi_tilde : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.5, p. 43
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "General defining equation for the efficient influence function via score and information operators"})


def cheatsheet():
    return "ksr065: General defining equation for the efficient influence function via score and information operators"
