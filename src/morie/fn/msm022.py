"""Numbered display equation (5.3) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def mvsml_linear_mixed_models_eq_5_3(types, are, the, four, environments, Besides):
    """
    Numbered display equation (5.3) from MVSML chapter 5.

    Formula: types are in the four environments. Besides the line indicator (GID), environment information (Env) was also available in the data set, which was needed for implementing model (5.4). The adopted structure for the variance–covariance matrix between environments is \SigmaE = \sigma2 EGII and the resulting model is referred to as M2. Another explored model (M20) was obtained under the same speciﬁcation, with the difference that G was set equal to the identity matrix. Using the same validation scheme that was used in Example 1, the results for each of the 10 random partitions are shown in Table 5.2, in which, for illustrative purposes, model

    Parameters
    ----------
    types : array-like
        Input data.
    are : array-like
        Input data.
    the : array-like
        Input data.
    four : array-like
        Input data.
    environments : array-like
        Input data.
    Besides : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.3) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    types = np.atleast_1d(np.asarray(types, dtype=float))
    n = len(types)
    result = float(np.mean(types))
    se = float(np.std(types, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.3) from MVSML chapter 5."})


def cheatsheet():
    return "msm022: Numbered display equation (5.3) from MVSML chapter 5."
