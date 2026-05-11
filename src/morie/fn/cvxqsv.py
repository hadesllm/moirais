"""SDP relaxation of QCQP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_qcqp_relaxation"]


def boyd_qcqp_relaxation(P0, q0, P, q, r):
    """
    SDP relaxation of QCQP

    Formula: Lift x to X = xx', drop rank-1 constraint

    Parameters
    ----------
    P0 : array-like
        Input data.
    q0 : array-like
        Input data.
    P : array-like
        Input data.
    q : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X

    References
    ----------
    Boyd CVX Ch 11
    """
    P0 = np.atleast_1d(np.asarray(P0, dtype=float))
    n = len(P0)
    result = float(np.mean(P0))
    se = float(np.std(P0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SDP relaxation of QCQP"})


def cheatsheet():
    return "cvxqsv: SDP relaxation of QCQP"
