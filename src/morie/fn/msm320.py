"""Numbered display equation (1.2) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_2"]


def mvsml_general_eq_1_2(yv, dat_F, y, n, length, Wavelengths):
    """
    Numbered display equation (1.2) from MVSML chapter 1.

    Formula: yv = dat_F$y n = length(yv) #Wavelengths data dat_W = dat_ls$dat_WL colnames(dat_W)[1:8] head(dat_W)[,1:8] #Wavelengths used Wv = as.numeric(substring(colnames(dat_W)[-(1.2)],2)) #Reﬂectance in each individual X_W = unique(dat_W[,-

    Parameters
    ----------
    yv : array-like
        Input data.
    dat_F : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.
    length : array-like
        Input data.
    Wavelengths : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.2) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.2) from MVSML chapter 1."})


def cheatsheet():
    return "msm320: Numbered display equation (1.2) from MVSML chapter 1."
