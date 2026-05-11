"""Numbered display equation (5.3) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def mvsml_linear_mixed_models_eq_5_3(Illustrative, Examples, of, the, Univariate, LMM):
    """
    Numbered display equation (5.3) from MVSML chapter 5.

    Formula: Illustrative Examples of the Univariate LMM Example 1 To illustrate the performance of the LMM in a genomic prediction context doing the ﬁtting process with the sommer package, we considered a wheat data set that consisted of 500 markers measured for each line as the genomic information, and with 229 observations in total that registered grain yield (tons/ ha): 30 lines in four environments with one or two repetitions. 5.4 Illustrative Examples of the Univariate LMM 149 Table 5.1 Prediction performance of the GBLUP model (5.3, M1) and the model

    Parameters
    ----------
    Illustrative : array-like
        Input data.
    Examples : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.
    Univariate : array-like
        Input data.
    LMM : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.3) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    Illustrative = np.atleast_1d(np.asarray(Illustrative, dtype=float))
    n = len(Illustrative)
    result = float(np.mean(Illustrative))
    se = float(np.std(Illustrative, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.3) from MVSML chapter 5."})


def cheatsheet():
    return "msm017: Numbered display equation (5.3) from MVSML chapter 5."
