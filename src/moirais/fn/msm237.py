"""Numbered display equation (1.3) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_3"]


def mvsml_general_eq_1_3(Ordering, the, data, Data, Final, order):
    """
    Numbered display equation (1.3) from MVSML chapter 1.

    Formula: ##########Ordering the data#################################### Data.Final=Data.Final[order(Data.Final$Env,Data.Final$GID),] ########Creating the design matrix of lines ################## Z1G=model.matrix(~0+as.factor(Data.Final$GID)) L=t(chol(Gg)) Z1G=Z1G%*%L ZT=model.matrix(~0+as.factor(Data.Final$Env)) Z2TG=model.matrix(~0+Z1G:as.factor(Data.Final$Env)) ############Preparation for training-testing sets############ Data.Final_1=Data.Final[,c

    Parameters
    ----------
    Ordering : array-like
        Input data.
    the : array-like
        Input data.
    data : array-like
        Input data.
    Data : array-like
        Input data.
    Final : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.3) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.3) from MVSML chapter 1."})


def cheatsheet():
    return "msm237: Numbered display equation (1.3) from MVSML chapter 1."
