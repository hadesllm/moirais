"""Numbered display equation (7.6) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_6"]


def mvsml_bayesian_regression_pt2_eq_7_6(tation, of, the, following, six, kinds):
    """
    Numbered display equation (7.6) from MVSML chapter 7.

    Formula: tation of the following six kinds of “GBLUP” models: GMLRM-R1, GMLRM-R2, GMLRM-R3, GMLRM-L1, GMLRM-L2, and GMLRM-L3. To evaluate the per- formance of these models, the same CV strategy was used, where for each of the 10 random partitions, 80% of the full data set was taken to train the models and the rest to evaluate their performance. 7.4 Penalized Multinomial Logistic Regression 231 Table 7.5 Brier score (BS) and proportion of cases correctly classiﬁed (PCCC) across 10 random partitions, with 80% of the data used for training and the rest for testing, for multinomial model

    Parameters
    ----------
    tation : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.
    following : array-like
        Input data.
    six : array-like
        Input data.
    kinds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.6) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    tation = np.atleast_1d(np.asarray(tation, dtype=float))
    n = len(tation)
    result = float(np.mean(tation))
    se = float(np.std(tation, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.6) from MVSML chapter 7."})


def cheatsheet():
    return "msm121: Numbered display equation (7.6) from MVSML chapter 7."
