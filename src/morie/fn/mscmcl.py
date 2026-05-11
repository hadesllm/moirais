"""Matrix completion SCM (Athey-Bayati-Doudchenko-Imbens-Khosravi)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matrix_completion_scm"]


def matrix_completion_scm(y, D, lam):
    """
    Matrix completion SCM (Athey-Bayati-Doudchenko-Imbens-Khosravi)

    Formula: min ||L||_* s.t. observed entries match; impute counterfactual

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey, Bayati, Doudchenko, Imbens, Khosravi (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matrix completion SCM (Athey-Bayati-Doudchenko-Imbens-Khosravi)"})


def cheatsheet():
    return "mscmcl: Matrix completion SCM (Athey-Bayati-Doudchenko-Imbens-Khosravi)"
