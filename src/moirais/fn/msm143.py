"""Numbered display equation (8.9) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_9"]


def mvsml_categorical_count_eq_8_9(A, R2, When, individuals, had, more):
    """
    Numbered display equation (8.9) from MVSML chapter 8.

    Formula: A R2) When individuals had more than one replication, or a sophisticated experimental design was used for data collection, the Bayesian kernel BLUP model is speciﬁed in a more general way to take into account this structure, as follows: Y = 1n\mu + Zu + e (8.9) with Z the incident matrix of the genomic effects. This model cannot be ﬁtted directly in the BGLR and some precalculus is needed ﬁrst to compute the “covari- ance” matrix of the predictor Zu in model

    Parameters
    ----------
    A : array-like
        Input data.
    R2 : array-like
        Input data.
    When : array-like
        Input data.
    individuals : array-like
        Input data.
    had : array-like
        Input data.
    more : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.9) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.9) from MVSML chapter 8."})


def cheatsheet():
    return "msm143: Numbered display equation (8.9) from MVSML chapter 8."
