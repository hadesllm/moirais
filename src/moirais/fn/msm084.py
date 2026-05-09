"""Numbered display equation (1.2) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_2"]


def mvsml_general_eq_1_2(Bayesian, Genomic, Linear, Regression, R, Code):
    """
    Numbered display equation (1.2) from MVSML chapter 1.

    Formula: 6 Bayesian Genomic Linear Regression R Code for Example 4 rm(list=ls(all=TRUE)) library(BMTME) library(dplyr) load('dat_ls.RData',verbose=TRUE) dat_F = dat_ls$dat_F head(dat_F) Y = as.matrix(dat_F[,-

    Parameters
    ----------
    Bayesian : array-like
        Input data.
    Genomic : array-like
        Input data.
    Linear : array-like
        Input data.
    Regression : array-like
        Input data.
    R : array-like
        Input data.
    Code : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.2) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    Bayesian = np.atleast_1d(np.asarray(Bayesian, dtype=float))
    n = len(Bayesian)
    result = float(np.mean(Bayesian))
    se = float(np.std(Bayesian, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.2) from MVSML chapter 1."})


def cheatsheet():
    return "msm084: Numbered display equation (1.2) from MVSML chapter 1."
