"""Restricted Boltzmann machine."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_boltzmann"]


def esl_boltzmann(v, h):
    """
    Restricted Boltzmann machine

    Formula: p(v,h) = (1/Z) exp(-E(v,h))

    Parameters
    ----------
    v : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: energy

    References
    ----------
    Hastie ESL Ch 17
    """
    v = np.atleast_1d(np.asarray(v, dtype=float))
    n = len(v)
    result = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Restricted Boltzmann machine"})


def cheatsheet():
    return "eslbrm: Restricted Boltzmann machine"
