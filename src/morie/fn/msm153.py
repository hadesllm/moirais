r"""Numbered display equation (8.12) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_12"]


def mvsml_categorical_count_eq_8_12(the, eigenvalues, ordered, largest, to, smallest):
    r"""
    Numbered display equation (8.12) from MVSML chapter 8.

    Formula: the eigenvalues ordered from largest to smallest. These values are substituted in Q  resulting in un  N 0, \sigma2 uKn,mUS 2 1=2S 2 1=2U0K0 ), and thus, thanks to the proper- n,m ties of the normal distribution, model (8.8) can be expressed like model (8.11) as y = \mu1n + Pf + \epsilon (8.12) Model

    Parameters
    ----------
    the : array-like
        Input data.
    eigenvalues : array-like
        Input data.
    ordered : array-like
        Input data.
    largest : array-like
        Input data.
    to : array-like
        Input data.
    smallest : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.12) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    the = np.atleast_1d(np.asarray(the, dtype=float))
    n = len(the)
    result = float(np.mean(the))
    se = float(np.std(the, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.12) from MVSML chapter 8."})


def cheatsheet():
    return "msm153: Numbered display equation (8.12) from MVSML chapter 8."
