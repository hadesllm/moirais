r"""Numbered display equation (8.11) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_11"]


def mvsml_categorical_count_eq_8_11(of, kernel, methods, However, this, section):
    r"""
    Numbered display equation (8.11) from MVSML chapter 8.

    Formula: of kernel methods. However, in this section, we only illustrate the method proposed by Cuevas et al. (2020). The basic idea of this method consists of approximating the original kernel using a small size (m) of the original n observations (lines in the context of GS) available in the training set, which signiﬁcantly reduces the required computational resources required to build the kernel matrix. Before giving the details of the compression of kernels proposed by Cuevas et al. (2020), it is important to point out that model (8.8) can be reparametrized as Eq. (8.11) if the eigenvalue decomposition of the kernel matrix K is expressed as US1/2S1/2U0, y = \mu1n + Pf + \epsilon,

    Parameters
    ----------
    of : array-like
        Input data.
    kernel : array-like
        Input data.
    methods : array-like
        Input data.
    However : array-like
        Input data.
    this : array-like
        Input data.
    section : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.11) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    of = np.atleast_1d(np.asarray(of, dtype=float))
    n = len(of)
    result = float(np.mean(of))
    se = float(np.std(of, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.11) from MVSML chapter 8."})


def cheatsheet():
    return "msm145: Numbered display equation (8.11) from MVSML chapter 8."
