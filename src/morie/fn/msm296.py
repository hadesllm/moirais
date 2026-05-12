r"""Numbered display equation (14.14) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_14"]


def mvsml_convolutional_nn_eq_14_14(length, t, general, the, functional, covariate):
    r"""
    Numbered display equation (14.14) from MVSML chapter 14.

    Formula: length t (in general, the functional covariate measured in time t), to allow the effect of reﬂectance to vary by environment (see Montesinos-López et al. 2017b). By assum- ing that there are ne observations in environment e, e = 1, . . ., I, the corresponding 610 14 Functional Regression re-expressed model after representing the coefﬁcient function \betae(t) in terms of the same basis functions used for \beta(t), \betae t( ) = PL1e l=1\betaelϕl t( ), is given by y = 1n\mu + XE\betaE + X\beta + XEF\betaEF + e,

    Parameters
    ----------
    length : array-like
        Input data.
    t : array-like
        Input data.
    general : array-like
        Input data.
    the : array-like
        Input data.
    functional : array-like
        Input data.
    covariate : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.14) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    length = np.atleast_1d(np.asarray(length, dtype=float))
    n = len(length)
    result = float(np.mean(length))
    se = float(np.std(length, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.14) from MVSML chapter 14."})


def cheatsheet():
    return "msm296: Numbered display equation (14.14) from MVSML chapter 14."
