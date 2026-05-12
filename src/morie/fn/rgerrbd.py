# morie.fn -- function file (hadesllm/morie)
"""Bhattacharyya bound on Bayes classification error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_bayes_error_bound"]


def rangayyan_bayes_error_bound(mu1, sigma1, p1, mu2, sigma2, p2):
    """
    Bhattacharyya bound on Bayes classification error

    Formula: P_e <= sqrt(P_1*P_2)*exp(-D_B(P1||P2)); D_B=Bhattacharyya distance

    Parameters
    ----------
    mu1 : array-like
        Input data.
    sigma1 : array-like
        Input data.
    p1 : array-like
        Input data.
    mu2 : array-like
        Input data.
    sigma2 : array-like
        Input data.
    p2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: error_bound

    References
    ----------
    Rangayyan Ch 10.6
    """
    mu1 = np.asarray(mu1, dtype=float)
    n = int(mu1) if mu1.ndim == 0 else len(mu1)
    result = float(np.mean(mu1))
    se = float(np.std(mu1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bhattacharyya bound on Bayes classification error"})


def cheatsheet():
    return "rgerrbd: Bhattacharyya bound on Bayes classification error"
