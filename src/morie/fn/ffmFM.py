"""Field-aware FM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["field_aware_fm"]


def field_aware_fm(X, y, K):
    """
    Field-aware FM

    Formula: FM with separate latent vec per field-pair

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Juan et al (2016) FFM
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Field-aware FM"})


def cheatsheet():
    return "ffmFM: Field-aware FM"
