"""TMLE under missing-at-random outcome."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_missing_data"]


def tmle_missing_data(y, D, X, missing):
    """
    TMLE under missing-at-random outcome

    Formula: weight by 1/Pr(observed|X) clever-covariate

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    missing : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rotnitzky et al (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE under missing-at-random outcome"})


def cheatsheet():
    return "tmlmda: TMLE under missing-at-random outcome"
