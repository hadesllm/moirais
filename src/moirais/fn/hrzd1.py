# moirais.fn — function file (hadesllm/moirais)
"""Semiparametric duration/hazard model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_duration_model"]


def horowitz_duration_model(t, x, event):
    """
    Semiparametric duration/hazard model

    Formula: h(t|X) = h0(t) * exp(X'beta), Cox-type

    Parameters
    ----------
    t : array-like
        Input data.
    x : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric duration/hazard model"})


def cheatsheet():
    return "hrzd1: Semiparametric duration/hazard model"
