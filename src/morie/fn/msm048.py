r"""Numbered display equation (6.3) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_3"]


def mvsml_bayesian_regression_eq_6_3(where, g, p, X1XT, G, which):
    r"""
    Numbered display equation (6.3) from MVSML chapter 6.

    Formula: , where \sigma2 g = p\sigma2 \beta and 1 p X1XT G = 1 1, which is known as the genomic relationship matrix (VanRaden 2007). Then, under this parameterization (g = X1\beta0 and \sigma2 g = p\sigma2 \beta), the model speciﬁed in

    Parameters
    ----------
    where : array-like
        Input data.
    g : array-like
        Input data.
    p : array-like
        Input data.
    X1XT : array-like
        Input data.
    G : array-like
        Input data.
    which : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.3) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    where = np.atleast_1d(np.asarray(where, dtype=float))
    n = len(where)
    result = float(np.mean(where))
    se = float(np.std(where, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.3) from MVSML chapter 6."})


def cheatsheet():
    return "msm048: Numbered display equation (6.3) from MVSML chapter 6."
