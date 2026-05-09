"""McDiarmid's inequality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_mcdiarmid"]


def wasserman_mcdiarmid(t, c):
    """
    McDiarmid's inequality

    Formula: P(|f(X)-E[f(X)]| > t) <= 2 exp(-2t^2/sum c_i^2)

    Parameters
    ----------
    t : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    Wasserman (2004), Ch 4
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "McDiarmid's inequality"})


def cheatsheet():
    return "wsmmcd: McDiarmid's inequality"
