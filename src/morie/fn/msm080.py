r"""Numbered display equation (6.11) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_11"]


def mvsml_bayesian_regression_eq_6_11(Return, to, step, terminate, when, chain):
    r"""
    Numbered display equation (6.11) from MVSML chapter 6.

    Formula: 8. Return to step 1 or terminate when chain length is adequate to meet convergence diagnostics and the required sample size is reached. A similar Gibbs sampler is implemented in the BMTME R package, with the main difference, that this package does not allow specifying a general ﬁxed effect design matrix X, only the corresponding to the design matrix for the environment effects, and also the intercept vector \mu is ignored because it is included in the ﬁxed 6.9 Bayesian Genomic Multi-trait and Multi-environment Model (BMTME) 197 environment effects. Speciﬁcally, to ﬁt model

    Parameters
    ----------
    Return : array-like
        Input data.
    to : array-like
        Input data.
    step : array-like
        Input data.
    terminate : array-like
        Input data.
    when : array-like
        Input data.
    chain : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.11) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    Return = np.atleast_1d(np.asarray(Return, dtype=float))
    n = len(Return)
    result = float(np.mean(Return))
    se = float(np.std(Return, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.11) from MVSML chapter 6."})


def cheatsheet():
    return "msm080: Numbered display equation (6.11) from MVSML chapter 6."
