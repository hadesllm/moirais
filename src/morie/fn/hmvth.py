# morie.fn — function file (hadesllm/morie)
"""Hard voting classifier: majority class vote among base models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_voting_hard"]


def geron_voting_hard(models, X):
    """
    Hard voting classifier: majority class vote among base models

    Formula: y_hat = mode_k({f_m(x)})

    Parameters
    ----------
    models : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Géron Ch 6
    """
    models = np.atleast_1d(np.asarray(models, dtype=float))
    n = len(models)
    result = float(np.mean(models))
    se = float(np.std(models, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hard voting classifier: majority class vote among base models"})


def cheatsheet():
    return "hmvth: Hard voting classifier: majority class vote among base models"
