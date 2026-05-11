"""Numbered display equation (5.5) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_5"]


def mvsml_linear_mixed_models_eq_5_5(similarly, x1, x2, xp, are, the):
    """
    Numbered display equation (5.5) from MVSML chapter 5.

    Formula: and similarly, x1, x2,. . ., xp are the column names of p covariates to be included in the ﬁtting process (see below the R code for Example 3). The rest of the arguments are the same as those described in the R code of model 5.3. 154 5 Linear Mixed Models Example 3 To illustrate the ﬁtting of the multi-trait genomic model (5.5) (M3), we considered the same data set used in Examples 1 and 2, but with the addition of trait (y2) to be able to explore the implementation of a bivariate trait genomic model. The same CV strategy implemented in Example 1 was used. In addition to model

    Parameters
    ----------
    similarly : array-like
        Input data.
    x1 : array-like
        Input data.
    x2 : array-like
        Input data.
    xp : array-like
        Input data.
    are : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.5) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    similarly = np.atleast_1d(np.asarray(similarly, dtype=float))
    n = len(similarly)
    result = float(np.mean(similarly))
    se = float(np.std(similarly, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.5) from MVSML chapter 5."})


def cheatsheet():
    return "msm031: Numbered display equation (5.5) from MVSML chapter 5."
