"""Numbered display equation (1.2) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_2"]


def mvsml_general_eq_1_2(Appendix, R, Code, Example, rm, list):
    """
    Numbered display equation (1.2) from MVSML chapter 1.

    Formula: Appendix 5 R Code Example 3 rm(list=ls(all=TRUE)) library(BGLR) library(BMTME) library(dplyr) load('dat_ls.RData',verbose=TRUE) dat_F = dat_ls$dat_F head(dat_F) Y = as.matrix(dat_F[,-

    Parameters
    ----------
    Appendix : array-like
        Input data.
    R : array-like
        Input data.
    Code : array-like
        Input data.
    Example : array-like
        Input data.
    rm : array-like
        Input data.
    list : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.2) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    Appendix = np.atleast_1d(np.asarray(Appendix, dtype=float))
    n = len(Appendix)
    result = float(np.mean(Appendix))
    se = float(np.std(Appendix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.2) from MVSML chapter 1."})


def cheatsheet():
    return "msm083: Numbered display equation (1.2) from MVSML chapter 1."
