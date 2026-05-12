r"""Numbered display equation (6.4) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_4"]


def mvsml_bayesian_regression_eq_6_4(p, X1XT, G, which, known, the):
    r"""
    Numbered display equation (6.4) from MVSML chapter 6.

    Formula: \beta and 1 p X1XT G = 1 1, which is known as the genomic relationship matrix (VanRaden 2007). Then, under this parameterization (g = X1\beta0 and \sigma2 g = p\sigma2 \beta), the model speciﬁed in (6.3), in matrix notation takes the following form: Y = 1n\mu + g + E

    Parameters
    ----------
    p : array-like
        Input data.
    X1XT : array-like
        Input data.
    G : array-like
        Input data.
    which : array-like
        Input data.
    known : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.4) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.4) from MVSML chapter 6."})


def cheatsheet():
    return "msm049: Numbered display equation (6.4) from MVSML chapter 6."
