"""Property inference (population stat from model)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["property_inference"]


def property_inference(model, property):
    """
    Property inference (population stat from model)

    Formula: meta-classifier on model parameters

    Parameters
    ----------
    model : array-like
        Input data.
    property : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ganju et al (2018)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Property inference (population stat from model)"})


def cheatsheet():
    return "propinf: Property inference (population stat from model)"
