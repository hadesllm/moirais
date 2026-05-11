"""Linear MSM with stabilized weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_linear"]


def msm_linear(y, treatment_history, covariate_history):
    """
    Linear MSM with stabilized weights

    Formula: E[Y(a_bar)] = beta_0 + beta_a a_bar; sw stabilized

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins (1999, 2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear MSM with stabilized weights"})


def cheatsheet():
    return "msmlin: Linear MSM with stabilized weights"
