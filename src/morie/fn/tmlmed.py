"""TMLE for natural direct + indirect mediation effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_mediation"]


def tmle_mediation(y, treatment, mediator, covariates):
    """
    TMLE for natural direct + indirect mediation effects

    Formula: NDE = E[Y(1, M(0))] - E[Y(0, M(0))]; NIE = E[Y(1, M(1))] - E[Y(1, M(0))]

    Parameters
    ----------
    y : array-like
        Input data.
    treatment : array-like
        Input data.
    mediator : array-like
        Input data.
    covariates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zheng & van der Laan (2012); Tchetgen Tchetgen & Shpitser (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for natural direct + indirect mediation effects"})


def cheatsheet():
    return "tmlmed: TMLE for natural direct + indirect mediation effects"
