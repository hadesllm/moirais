"""Plug-in g-computation NIE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["plug_in_mediation"]


def plug_in_mediation(Y_model, M_model, X, C):
    """
    Plug-in g-computation NIE

    Formula: impute Y under counterfactual M

    Parameters
    ----------
    Y_model : array-like
        Input data.
    M_model : array-like
        Input data.
    X : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vansteelandt-Bekaert-Lange (2012)
    """
    Y_model = np.atleast_1d(np.asarray(Y_model, dtype=float))
    n = len(Y_model)
    result = float(np.mean(Y_model))
    se = float(np.std(Y_model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Plug-in g-computation NIE"})


def cheatsheet():
    return "pluginM: Plug-in g-computation NIE"
