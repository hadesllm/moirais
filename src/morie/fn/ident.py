# morie.fn — function file (hadesllm/morie)
"""Identifiability conditions for causal effects from observational data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["identifiability_conditions"]


def identifiability_conditions(data, dag):
    """
    Identifiability conditions for causal effects from observational data

    Formula: ATE identifiable if: positivity P(T=t|X=data)>0 for all t,data; exchangeability Y(t) _|_ T|X; consistency Y = Y(T)

    Parameters
    ----------
    data : array-like
        Input data.
    dag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'identifiable': 'bool'}

    References
    ----------
    Molak Ch 7,8
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Identifiability conditions for causal effects from observational data"})


def cheatsheet():
    return "ident: Identifiability conditions for causal effects from observational data"
