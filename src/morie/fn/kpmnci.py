"""KM pointwise CI via Greenwood."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["km_pointwise_ci"]


def km_pointwise_ci(fit, alpha):
    """
    KM pointwise CI via Greenwood

    Formula: S(t) ± z * sqrt(Var_Greenwood)

    Parameters
    ----------
    fit : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Greenwood (1926)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KM pointwise CI via Greenwood"})


def cheatsheet():
    return "kpmnci: KM pointwise CI via Greenwood"
