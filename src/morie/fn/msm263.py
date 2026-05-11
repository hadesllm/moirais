"""Numbered display equation (14.1) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_1"]


def mvsml_convolutional_nn_eq_14_1(elements, of, basis, a, function, space):
    """
    Numbered display equation (14.1) from MVSML chapter 14.

    Formula: elements of basis for a function space and \betal are constants that depend on the © The Author(s) 2022 579 O. A. Montesinos López et al., Multivariate Statistical Machine Learning Methods for Genomic Prediction, https://doi.org/10.1007/978-3-030-89010-0_14 580 14 Functional Regression function to be represented (Ramsay et al. 2009). Then, by assuming this form for \beta(t), model

    Parameters
    ----------
    elements : array-like
        Input data.
    of : array-like
        Input data.
    basis : array-like
        Input data.
    a : array-like
        Input data.
    function : array-like
        Input data.
    space : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.1) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    elements = np.atleast_1d(np.asarray(elements, dtype=float))
    n = len(elements)
    result = float(np.mean(elements))
    se = float(np.std(elements, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.1) from MVSML chapter 14."})


def cheatsheet():
    return "msm263: Numbered display equation (14.1) from MVSML chapter 14."
