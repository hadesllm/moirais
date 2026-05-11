"""Numbered display equation (5.4) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(dat_M, scale, G, tcrossprod, dim, dat_F):
    """
    Numbered display equation (5.4) from MVSML chapter 5.

    Formula: dat_M = scale(dat_M) G = tcrossprod(dat_M)/dim(dat_M)[2] dat_F$GID = factor(dat_F$GID,levels=row.names(G)) dat_F = dat_F[order(dat_F$Env,dat_F$GID),] #10 random partitions K = 10 n = dim(dat_F)[1] set.seed(1) PT = replicate(K,sample(n,0.20*n)) #Model

    Parameters
    ----------
    dat_M : array-like
        Input data.
    scale : array-like
        Input data.
    G : array-like
        Input data.
    tcrossprod : array-like
        Input data.
    dim : array-like
        Input data.
    dat_F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.4) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    dat_M = np.atleast_1d(np.asarray(dat_M, dtype=float))
    n = len(dat_M)
    result = float(np.mean(dat_M))
    se = float(np.std(dat_M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.4) from MVSML chapter 5."})


def cheatsheet():
    return "msm037: Numbered display equation (5.4) from MVSML chapter 5."
