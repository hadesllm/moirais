r"""Numbered display equation (15.2) from MVSML chapter 15.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_functional_regression_eq_15_2"]


def mvsml_functional_regression_eq_15_2(log, Y, i):
    r"""
    Numbered display equation (15.2) from MVSML chapter 15.

    Formula: log Y+ ( ( ) ) + log \mu ( ) i ! , i i

    Parameters
    ----------
    log : array-like
        Input data.
    Y : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (15.2) [Multivariate Statistical Machine Learnin [Pages 633-681] [2026-04-16].pdf]
    r"""
    log = np.atleast_1d(np.asarray(log, dtype=float))
    n = len(log)
    result = float(np.mean(log))
    se = float(np.std(log, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (15.2) from MVSML chapter 15."})


def cheatsheet():
    return "msm325: Numbered display equation (15.2) from MVSML chapter 15."
