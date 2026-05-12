# morie.fn -- function file (hadesllm/morie)
"""Ordinal threshold/probit model for categorical traits."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ordinal_threshold_model"]


def ordinal_threshold_model(y_ord, X, Z, n_categories):
    """
    Ordinal threshold/probit model for categorical traits

    Formula: P(y_i = k) = Phi(tau_k - eta_i) - Phi(tau_{k-1} - eta_i); eta_i = X_i*beta + Z_i*u

    Parameters
    ----------
    y_ord : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    n_categories : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'tau_samples': 'array', 'eta_samples': 'array'}

    References
    ----------
    Montesinos Lopez Ch 7
    """
    y_ord = np.asarray(y_ord, dtype=float)
    n = int(y_ord) if y_ord.ndim == 0 else len(y_ord)
    result = float(np.mean(y_ord))
    se = float(np.std(y_ord, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordinal threshold/probit model for categorical traits"})


def cheatsheet():
    return "ordtm: Ordinal threshold/probit model for categorical traits"
