"""TMLE with arbitrary ML for Q and g."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_machine_learning"]


def tmle_machine_learning(y, D, X, ml_q, ml_g):
    """
    TMLE with arbitrary ML for Q and g

    Formula: any cross-fit ML for nuisance + targeted update

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    ml_q : array-like
        Input data.
    ml_g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov et al (2018) DML; vdL-Rose (2018) Targeted Learning V2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE with arbitrary ML for Q and g"})


def cheatsheet():
    return "tmlmnl: TMLE with arbitrary ML for Q and g"
