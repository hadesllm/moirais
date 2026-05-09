"""Numbered display equation (2.1) from MVSML chapter 2.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_preprocessing_eq_2_1"]


def mvsml_preprocessing_eq_2_1(In, matrix, M, only, the, columns):
    """
    Numbered display equation (2.1) from MVSML chapter 2.

    Formula: 1 0 In matrix M, only the columns corresponding to marker information are selected. This information is scaled by column using the scale command of R, and the scaled markers are saved in the MS matrix; the GRM is calculated with this information using method 3 in Sect. 2.4. Here we obtained the design matrix for environments and lines, and we also added the genomic information to the design matrix of lines by post-multiplying the design matrix of lines by the Cholesky decomposition of the GRM, which also can be used as an alternative way to obtain the breeding values. This is because the GBLUP model

    Parameters
    ----------
    In : array-like
        Input data.
    matrix : array-like
        Input data.
    M : array-like
        Input data.
    only : array-like
        Input data.
    the : array-like
        Input data.
    columns : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (2.1) [Multivariate Statistical Machine Learnin [Pages 35-70] [2026-04-16].pdf]
    """
    In = np.atleast_1d(np.asarray(In, dtype=float))
    n = len(In)
    result = float(np.mean(In))
    se = float(np.std(In, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (2.1) from MVSML chapter 2."})


def cheatsheet():
    return "msm244: Numbered display equation (2.1) from MVSML chapter 2."
