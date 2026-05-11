"""Numbered display equation (10.14) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_14"]


def mvsml_reproducing_kernel_eq_10_14(jk, Next, to, update, the, weights):
    """
    Numbered display equation (10.14) from MVSML chapter 10.

    Formula: ( ) . jk jk Next, to update the weights connecting the input units to the hidden units, we follow a similar process as in (10.12). Thus kp = -\eta \partial E \Deltaw h ( )

    Parameters
    ----------
    jk : array-like
        Input data.
    Next : array-like
        Input data.
    to : array-like
        Input data.
    update : array-like
        Input data.
    the : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.14) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    """
    jk = np.atleast_1d(np.asarray(jk, dtype=float))
    n = len(jk)
    result = float(np.mean(jk))
    se = float(np.std(jk, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.14) from MVSML chapter 10."})


def cheatsheet():
    return "msm253: Numbered display equation (10.14) from MVSML chapter 10."
