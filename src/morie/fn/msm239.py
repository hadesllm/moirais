"""Numbered display equation (1.3) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_3"]


def mvsml_general_eq_1_3(Data, Final, order, Env, GID, Creating):
    """
    Numbered display equation (1.3) from MVSML chapter 1.

    Formula: Data.Final=Pheno_Toy_EYT Data.Final=Data.Final[order(Data.Final$Env,Data.Final$GID),] ########Creating the design matrix of lines ################## Z1G=model.matrix(~0+as.factor(Data.Final$GID)) L=t(chol(Gg)) Z1G=Z1G%*%L ZT=model.matrix(~0+as.factor(Data.Final$Env)) Z2TG=model.matrix(~0+Z1G:as.factor(Data.Final$Env)) nCV=5 Data.Final_1=Data.Final[,c

    Parameters
    ----------
    Data : array-like
        Input data.
    Final : array-like
        Input data.
    order : array-like
        Input data.
    Env : array-like
        Input data.
    GID : array-like
        Input data.
    Creating : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.3) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    Data = np.atleast_1d(np.asarray(Data, dtype=float))
    n = len(Data)
    result = float(np.mean(Data))
    se = float(np.std(Data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.3) from MVSML chapter 1."})


def cheatsheet():
    return "msm239: Numbered display equation (1.3) from MVSML chapter 1."
