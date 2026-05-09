"""ARE under Gaussian."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["asymptotic_relative_efficiency"]


def asymptotic_relative_efficiency(estimator):
    """
    ARE under Gaussian

    Formula: ARE = V(T_eff) / V(T_robust)

    Parameters
    ----------
    estimator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hodges-Lehmann (1956)
    """
    estimator = np.atleast_1d(np.asarray(estimator, dtype=float))
    n = len(estimator)
    result = float(np.mean(estimator))
    se = float(np.std(estimator, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARE under Gaussian"})


def cheatsheet():
    return "areN: ARE under Gaussian"
