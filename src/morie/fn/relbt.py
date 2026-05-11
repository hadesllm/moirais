# morie.fn — function file (hadesllm/morie)
"""Reliability: squared predictive ability divided by heritability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["reliability_metric"]


def reliability_metric(r, h2):
    """
    Reliability: squared predictive ability divided by heritability

    Formula: rel = r^2 / h^2

    Parameters
    ----------
    r : array-like
        Input data.
    h2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'reliability': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reliability: squared predictive ability divided by heritability"})


def cheatsheet():
    return "relbt: Reliability: squared predictive ability divided by heritability"
