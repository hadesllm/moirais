"""Numbered display equation (1.2) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_2"]


def mvsml_general_eq_1_2(load, dat_ls, RData, Phenotypic, data, dat_F):
    """
    Numbered display equation (1.2) from MVSML chapter 1.

    Formula: load('dat_ls.RData') #Phenotypic data dat_F = dat_ls$dat_F head(dat_F) #Wavelengths data dat_W = dat_ls$dat_WL colnames(dat_W)[1:8] head(dat_W)[,1:8] #Wavelengths used Wv = as.numeric(substring(colnames(dat_W)[-

    Parameters
    ----------
    load : array-like
        Input data.
    dat_ls : array-like
        Input data.
    RData : array-like
        Input data.
    Phenotypic : array-like
        Input data.
    data : array-like
        Input data.
    dat_F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.2) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.2) from MVSML chapter 1."})


def cheatsheet():
    return "msm315: Numbered display equation (1.2) from MVSML chapter 1."
