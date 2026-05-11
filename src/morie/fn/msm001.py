"""Numbered display equation (1.1) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_1"]


def mvsml_general_eq_1_1(A, model, a, simpli, ed, description):
    """
    Numbered display equation (1.1) from MVSML chapter 1.

    Formula: A model is a simpliﬁed description, using mathematical tools, of the processes we think that give rise to the observations in a set of data. A model is deterministic if it explains (completely) the dependent variables based on the independent ones. In many real-world scenarios, this is not possible. Instead, statistical (or stochastic) models try to approximate exact solutions by evaluating probabilistic distributions. For this reason, a statistical model is expressed by an equation composed of a systematic (deterministic) and a random part (Stroup 2012) as given in the next equation: yi = f xi ( ) + Ei, for i = 1, 2, . . . , n,

    Parameters
    ----------
    A : array-like
        Input data.
    model : array-like
        Input data.
    a : array-like
        Input data.
    simpli : array-like
        Input data.
    ed : array-like
        Input data.
    description : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.1) [Multivariate Statistical Machine Learnin [Pages 1-34] [2026-04-16].pdf]
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.1) from MVSML chapter 1."})


def cheatsheet():
    return "msm001: Numbered display equation (1.1) from MVSML chapter 1."
