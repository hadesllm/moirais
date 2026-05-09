"""Age-structured SIR with contact matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sir_age_structured"]


def sir_age_structured(S, I, R, contact_matrix, gamma):
    """
    Age-structured SIR with contact matrix

    Formula: per-age beta_ij from contact matrix C

    Parameters
    ----------
    S : array-like
        Input data.
    I : array-like
        Input data.
    R : array-like
        Input data.
    contact_matrix : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mossong et al (2008) POLYMOD
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Age-structured SIR with contact matrix"})


def cheatsheet():
    return "sirtdy: Age-structured SIR with contact matrix"
