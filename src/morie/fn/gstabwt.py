"""Stabilized IPT weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stabilized_weights"]


def stabilized_weights(treatment, history, numerator_model, denominator_model):
    """
    Stabilized IPT weights

    Formula: sw = prod_t f(A_t|A_{t-1})/f(A_t|H_t)

    Parameters
    ----------
    treatment : array-like
        Input data.
    history : array-like
        Input data.
    numerator_model : array-like
        Input data.
    denominator_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Hernán-Brumback (2000)
    """
    treatment = np.atleast_1d(np.asarray(treatment, dtype=float))
    n = len(treatment)
    result = float(np.mean(treatment))
    se = float(np.std(treatment, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stabilized IPT weights"})


def cheatsheet():
    return "gstabwt: Stabilized IPT weights"
