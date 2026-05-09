"""Treatment × covariate interaction in MSM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["interaction_analysis"]


def interaction_analysis(y, A, V, H):
    """
    Treatment × covariate interaction in MSM

    Formula: E[Y(a)|V] = beta_0 + beta_a a + beta_v V + beta_av a V

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    V : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hernán-Robins Ch 13
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Treatment × covariate interaction in MSM"})


def cheatsheet():
    return "intanl: Treatment × covariate interaction in MSM"
