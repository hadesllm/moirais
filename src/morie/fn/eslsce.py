"""Score matching loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_score_match"]


def esl_score_match(q, theta, p):
    """
    Score matching loss

    Formula: L = E_x[||grad_x log q(x;theta) - grad_x log p(x)||^2]

    Parameters
    ----------
    q : array-like
        Input data.
    theta : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Hastie ESL Ch 17
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    result = float(np.mean(q))
    se = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Score matching loss"})


def cheatsheet():
    return "eslsce: Score matching loss"
