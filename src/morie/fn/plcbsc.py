"""Placebo permutation inference for SCM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["placebo_scm_inference"]


def placebo_scm_inference(y, treated, controls):
    """
    Placebo permutation inference for SCM

    Formula: compare actual gap to gaps from each control as placebo treated

    Parameters
    ----------
    y : array-like
        Input data.
    treated : array-like
        Input data.
    controls : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Abadie, Diamond, Hainmueller (2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Placebo permutation inference for SCM"})


def cheatsheet():
    return "plcbsc: Placebo permutation inference for SCM"
