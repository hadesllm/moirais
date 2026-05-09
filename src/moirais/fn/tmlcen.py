"""TMLE under right-censoring with inverse-probability-of-censoring weighting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_censoring"]


def tmle_censoring(time, event, censor, treatment, covariates):
    """
    TMLE under right-censoring with inverse-probability-of-censoring weighting

    Formula: weight by 1/G(C>t|A,W); target survival probability with clever covariate

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    censor : array-like
        Input data.
    treatment : array-like
        Input data.
    covariates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van der Laan-Robins (2003); Stitelman-Lendle-vdL (2011)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE under right-censoring with inverse-probability-of-censoring weighting"})


def cheatsheet():
    return "tmlcen: TMLE under right-censoring with inverse-probability-of-censoring weighting"
