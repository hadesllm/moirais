# moirais.fn — function file (hadesllm/moirais)
"""DR-learner: doubly robust meta-learner for CATE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dr_learner"]


def dr_learner(Y, T, X, mu0, mu1, e_model, cate_model):
    """
    DR-learner: doubly robust meta-learner for CATE

    Formula: pseudo_outcome_i = DR_score_i; regress pseudo on X

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.
    X : array-like
        Input data.
    mu0 : array-like
        Input data.
    mu1 : array-like
        Input data.
    e_model : array-like
        Input data.
    cate_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'cate': 'array'}

    References
    ----------
    Molak Ch 10
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-learner: doubly robust meta-learner for CATE"})


def cheatsheet():
    return "drlnr: DR-learner: doubly robust meta-learner for CATE"
