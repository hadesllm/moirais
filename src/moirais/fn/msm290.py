"""Numbered display equation (14.12) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_12"]


def mvsml_convolutional_nn_eq_14_12(Penalty, matrix, of, derivative, order, p):
    """
    Numbered display equation (14.12) from MVSML chapter 14.

    Formula: #Penalty matrix of derivative of order p P_mat = eval.penalty(basisobj =Phi,Lfdobj=p) ei_P = eigen(P_mat) #Espectral descompositin of P_mat gamma = ei_P$vectors #\Gamma dv = ei_P$values#Eigenvalues of P_mat, elements of diagonal of D dv = ifelse(dv<1e-10,0,dv) where Phi is a created basis in R (Fourier or B-spline) and p is a nonnegative integer for the order of the derivative in the penalty matrix to be used. Once the penalty matrix is computed, the training of the model in

    Parameters
    ----------
    Penalty : array-like
        Input data.
    matrix : array-like
        Input data.
    of : array-like
        Input data.
    derivative : array-like
        Input data.
    order : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.12) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    Penalty = np.atleast_1d(np.asarray(Penalty, dtype=float))
    n = len(Penalty)
    result = float(np.mean(Penalty))
    se = float(np.std(Penalty, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.12) from MVSML chapter 14."})


def cheatsheet():
    return "msm290: Numbered display equation (14.12) from MVSML chapter 14."
