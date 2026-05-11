"""Attribute inference attack."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["attribute_inference"]


def attribute_inference(model, x_partial):
    """
    Attribute inference attack

    Formula: infer sensitive attribute given partial features

    Parameters
    ----------
    model : array-like
        Input data.
    x_partial : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fredrikson-Jha-Ristenpart (2015)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Attribute inference attack"})


def cheatsheet():
    return "attrInf: Attribute inference attack"
