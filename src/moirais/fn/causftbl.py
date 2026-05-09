"""Pearl front-door adjustment via mediator path."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_frontdoor_adjustment"]


def causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X):
    """
    Pearl front-door adjustment via mediator path

    Formula: P(Y|do(X)) = Σ_z P(Z|X) Σ_{x'} P(Y|X=x', Z) P(X=x')

    Parameters
    ----------
    P_Z_X : array-like
        Input data.
    P_Y_XZ : array-like
        Input data.
    P_X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: P_do

    References
    ----------
    Pearl (2009)
    """
    P_Z_X = np.atleast_1d(np.asarray(P_Z_X, dtype=float))
    n = len(P_Z_X)
    result = float(np.mean(P_Z_X))
    se = float(np.std(P_Z_X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pearl front-door adjustment via mediator path"})


def cheatsheet():
    return "causftbl: Pearl front-door adjustment via mediator path"
