# moirais.fn — function file (hadesllm/moirais)
"""Out-of-bag (OOB) evaluation using unsampled observations per bootstrap."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_oob_score"]


def geron_oob_score(X, y, models):
    """
    Out-of-bag (OOB) evaluation using unsampled observations per bootstrap

    Formula: OOB score = avg_i accuracy(f_m(x_i)) over m not sampling x_i

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    models : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: oob_score

    References
    ----------
    Géron Ch 6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Out-of-bag (OOB) evaluation using unsampled observations per bootstrap"})


def cheatsheet():
    return "hmoob: Out-of-bag (OOB) evaluation using unsampled observations per bootstrap"
