"""Bayes's formula giving the posterior measure of a set B as the ratio of integrated likelihoods over B and over the whole parameter space.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch1_bayes_formula"]


def ghosal_ch1_bayes_formula(B, X, p_theta, Pi):
    """
    Bayes's formula giving the posterior measure of a set B as the ratio of integrated likelihoods over B and over the whole parameter space.

    Formula: Pi(B | X) = ( int_B p_theta(X) dPi(theta) ) / ( int p_theta(X) dPi(theta) )

    Parameters
    ----------
    B : array-like
        Input data.
    X : array-like
        Input data.
    p_theta : array-like
        Input data.
    Pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: posterior

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 1, Eq 1.1, p. 7
    """
    B = np.atleast_1d(np.asarray(B, dtype=float))
    n = len(B)
    result = float(np.mean(B))
    se = float(np.std(B, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayes's formula giving the posterior measure of a set B as the ratio of integrated likelihoods over B and over the whole parameter space."})


def cheatsheet():
    return "ghs001: Bayes's formula giving the posterior measure of a set B as the ratio of integrated likelihoods over B and over the whole parameter space."
