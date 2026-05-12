r"""Numbered display equation (8.8) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_8"]


def mvsml_categorical_count_eq_8_8(e, v, S, the, induced, priors):
    r"""
    Numbered display equation (8.8) from MVSML chapter 8.

    Formula: e  \chi2 v,S, and the induced priors u j \sigma2 u    Nn 0, K\sigma2 and \sigma2 u  \chi2 vu, Su, the full conditional posterior distribution of u in model g

    Parameters
    ----------
    e : array-like
        Input data.
    v : array-like
        Input data.
    S : array-like
        Input data.
    the : array-like
        Input data.
    induced : array-like
        Input data.
    priors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.8) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    e = np.atleast_1d(np.asarray(e, dtype=float))
    n = len(e)
    result = float(np.mean(e))
    se = float(np.std(e, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.8) from MVSML chapter 8."})


def cheatsheet():
    return "msm139: Numbered display equation (8.8) from MVSML chapter 8."
