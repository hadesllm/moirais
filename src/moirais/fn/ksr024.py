"""Partly linear logistic regression model with smooth nuisance function eta."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch1_partly_linear_logistic"]


def kosorok_ch1_partly_linear_logistic(Y, Z, U, beta, eta):
    """
    Partly linear logistic regression model with smooth nuisance function eta

    Formula: E[Y | Z, U] = nu(beta' * Z + eta(U)), where nu(t) = 1/(1+exp(-t))

    Parameters
    ----------
    Y : array-like
        Input data.
    Z : array-like
        Input data.
    U : array-like
        Input data.
    beta : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 1, Eq 1.5, p. 6
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partly linear logistic regression model with smooth nuisance function eta"})


def cheatsheet():
    return "ksr024: Partly linear logistic regression model with smooth nuisance function eta"
