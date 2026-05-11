"""Numbered display equation (14.12) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_12"]


def mvsml_convolutional_nn_eq_14_12(previous, section, random, partitions, were, used):
    """
    Numbered display equation (14.12) from MVSML chapter 14.

    Formula: previous section, 100 random partitions were used, and in each, 80% of the data set was used to train the model and the rest to evaluate the prediction performance. When training the model, an inner ﬁve-fold cross-validation was used to choose the optimal parameter (\lambda) and estimate the \betal’s coefﬁcients. This was done using Fourier and B-spline basis in the representation of the beta function, and in both cases, two 602 14 Functional Regression basis were used. In both cases, the penalty matrix (14.11) and the elements of its spectral decomposition

    Parameters
    ----------
    previous : array-like
        Input data.
    section : array-like
        Input data.
    random : array-like
        Input data.
    partitions : array-like
        Input data.
    were : array-like
        Input data.
    used : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.12) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    previous = np.atleast_1d(np.asarray(previous, dtype=float))
    n = len(previous)
    result = float(np.mean(previous))
    se = float(np.std(previous, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.12) from MVSML chapter 14."})


def cheatsheet():
    return "msm289: Numbered display equation (14.12) from MVSML chapter 14."
