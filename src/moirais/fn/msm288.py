"""Numbered display equation (14.11) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_11"]


def mvsml_convolutional_nn_eq_14_11(accuracy, of, this, the, non, penalized):
    """
    Numbered display equation (14.11) from MVSML chapter 14.

    Formula: accuracy of this with the non-penalized functional regression described in the previous section, 100 random partitions were used, and in each, 80% of the data set was used to train the model and the rest to evaluate the prediction performance. When training the model, an inner ﬁve-fold cross-validation was used to choose the optimal parameter (\lambda) and estimate the \betal’s coefﬁcients. This was done using Fourier and B-spline basis in the representation of the beta function, and in both cases, two 602 14 Functional Regression basis were used. In both cases, the penalty matrix

    Parameters
    ----------
    accuracy : array-like
        Input data.
    of : array-like
        Input data.
    this : array-like
        Input data.
    the : array-like
        Input data.
    non : array-like
        Input data.
    penalized : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.11) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    accuracy = np.atleast_1d(np.asarray(accuracy, dtype=float))
    n = len(accuracy)
    result = float(np.mean(accuracy))
    se = float(np.std(accuracy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.11) from MVSML chapter 14."})


def cheatsheet():
    return "msm288: Numbered display equation (14.11) from MVSML chapter 14."
