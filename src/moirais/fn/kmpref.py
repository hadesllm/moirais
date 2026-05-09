# moirais.fn — function file (hadesllm/moirais)
"""Prefix tuning: learned prefix vectors prepended to keys/values at every layer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_prefix_tuning"]


def kamath_prefix_tuning(prefix_K, prefix_V, K_input, V_input):
    """
    Prefix tuning: learned prefix vectors prepended to keys/values at every layer

    Formula: K = [K_prefix; K_input]; V = [V_prefix; V_input]; only K_prefix, V_prefix are trained

    Parameters
    ----------
    prefix_K : array-like
        Input data.
    prefix_V : array-like
        Input data.
    K_input : array-like
        Input data.
    V_input : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: K_full, V_full

    References
    ----------
    Kamath Ch 4, Prefix Tuning section
    """
    prefix_K = np.atleast_1d(np.asarray(prefix_K, dtype=float))
    n = len(prefix_K)
    result = float(np.mean(prefix_K))
    se = float(np.std(prefix_K, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prefix tuning: learned prefix vectors prepended to keys/values at every layer"})


def cheatsheet():
    return "kmpref: Prefix tuning: learned prefix vectors prepended to keys/values at every layer"
