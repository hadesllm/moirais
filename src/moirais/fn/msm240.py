"""Numbered display equation (2.1) from MVSML chapter 2.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_preprocessing_eq_2_1"]


def mvsml_preprocessing_eq_2_1(represented, by, random, variables, observed, which):
    """
    Numbered display equation (2.1) from MVSML chapter 2.

    Formula: represented by random variables (not observed) which, we generally assume, have a particular distribution, the normal distribution being the most common. Due to the above, random effects are suggested when we want to perform an inference for all levels of the target population. 2.2 BLUEs and BLUPs This section presents the concepts and terminologies of BLUE and BLUP. Since these two concepts are related to a mixed model, we present the following linear mixed model as Y = X\beta + Zu + \epsilon,

    Parameters
    ----------
    represented : array-like
        Input data.
    by : array-like
        Input data.
    random : array-like
        Input data.
    variables : array-like
        Input data.
    observed : array-like
        Input data.
    which : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (2.1) [Multivariate Statistical Machine Learnin [Pages 35-70] [2026-04-16].pdf]
    """
    represented = np.atleast_1d(np.asarray(represented, dtype=float))
    n = len(represented)
    result = float(np.mean(represented))
    se = float(np.std(represented, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (2.1) from MVSML chapter 2."})


def cheatsheet():
    return "msm240: Numbered display equation (2.1) from MVSML chapter 2."
