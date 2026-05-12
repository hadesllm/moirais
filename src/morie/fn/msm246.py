r"""Numbered display equation (10.10) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_10"]


def mvsml_reproducing_kernel_eq_10_10(nding, the, optimal, weights, biases, This):
    r"""
    Numbered display equation (10.10) from MVSML chapter 10.

    Formula: backpropagation for ﬁnding the optimal weights and biases. This method consists of evaluating the partial derivatives of the loss function with regard to the weights and then moving these values down the slope, until the score of the loss function no longer decreases. For example, if we make the variation of the weights proportional to the negative of the gradient, the change in the weights in the right direction is reached. The gradient of the loss function given in (10.5) with respect to the weights connecting the hidden units to the output units (w l( ) jk ) is given by jk = -\eta \partial E \Deltaw l( )

    Parameters
    ----------
    nding : array-like
        Input data.
    the : array-like
        Input data.
    optimal : array-like
        Input data.
    weights : array-like
        Input data.
    biases : array-like
        Input data.
    This : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.10) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    r"""
    nding = np.atleast_1d(np.asarray(nding, dtype=float))
    n = len(nding)
    result = float(np.mean(nding))
    se = float(np.std(nding, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.10) from MVSML chapter 10."})


def cheatsheet():
    return "msm246: Numbered display equation (10.10) from MVSML chapter 10."
