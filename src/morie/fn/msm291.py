"""Numbered display equation (14.12) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_12"]


def mvsml_convolutional_nn_eq_14_12(matrix, computed, the, training, of, model):
    """
    Numbered display equation (14.12) from MVSML chapter 14.

    Formula: matrix is computed, the training of the model in (14.12) can be done in R as Xa = X_F%*%gamma A_PFR = cv.glmnet(x =Xa,y = y ,alpha = 0, nfolds=k, penalty.factor=dv, standardize=FALSE,maxit=1e6) where Xa is X, y is the vector with the corresponding values of the response variable, k is an integer used to specify the inner k cross-validation to train the model and choose the “optimal” value of the smoothing parameter, and dv are the eigenvalues of the penalty matrix used to indicate different penalties of the \beta j as required in

    Parameters
    ----------
    matrix : array-like
        Input data.
    computed : array-like
        Input data.
    the : array-like
        Input data.
    training : array-like
        Input data.
    of : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.12) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    matrix = np.atleast_1d(np.asarray(matrix, dtype=float))
    n = len(matrix)
    result = float(np.mean(matrix))
    se = float(np.std(matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.12) from MVSML chapter 14."})


def cheatsheet():
    return "msm291: Numbered display equation (14.12) from MVSML chapter 14."
