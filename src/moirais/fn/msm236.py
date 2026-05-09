"""Numbered display equation (1.3) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_3"]


def mvsml_general_eq_1_3(Creating, the, design, matrix, of, lines):
    """
    Numbered display equation (1.3) from MVSML chapter 1.

    Formula: ########Creating the design matrix of lines ################## Z1G=model.matrix(~0+as.factor(Data.Final$GID)) L=t(chol(Gg)) Z1G=Z1G%*%L ZT=model.matrix(~0+as.factor(Data.Final$Env)) Z2TG=model.matrix(~0+Z1G:as.factor(Data.Final$Env)) Then with the next part of the code, we prepare the information to create the folds for implementing a ﬁve-fold CV strategy. ##########Preparation for building the ﬁve-fold CV###### Data.Final_1=Data.Final[,c

    Parameters
    ----------
    Creating : array-like
        Input data.
    the : array-like
        Input data.
    design : array-like
        Input data.
    matrix : array-like
        Input data.
    of : array-like
        Input data.
    lines : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.3) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    Creating = np.atleast_1d(np.asarray(Creating, dtype=float))
    n = len(Creating)
    result = float(np.mean(Creating))
    se = float(np.std(Creating, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.3) from MVSML chapter 1."})


def cheatsheet():
    return "msm236: Numbered display equation (1.3) from MVSML chapter 1."
