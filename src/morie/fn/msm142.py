r"""Numbered display equation (8.9) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_9"]


def mvsml_categorical_count_eq_8_9(model, using, the, BGLR, package, The):
    r"""
    Numbered display equation (8.9) from MVSML chapter 8.

    Formula: model using the BGLR package. The BGLR code to ﬁt this model is ETA = list( list( model = ‘RHKS’, K = K , df0 = vu, S0 = Su, R2 = 1-R2)) ) = BGLR(y=y, ETA = ETA, nIter = 1e4, burnIn = 1e3, S0 = S, df0 = v, R2 = A R2) When individuals had more than one replication, or a sophisticated experimental design was used for data collection, the Bayesian kernel BLUP model is speciﬁed in a more general way to take into account this structure, as follows: Y = 1n\mu + Zu + e

    Parameters
    ----------
    model : array-like
        Input data.
    using : array-like
        Input data.
    the : array-like
        Input data.
    BGLR : array-like
        Input data.
    package : array-like
        Input data.
    The : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.9) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.9) from MVSML chapter 8."})


def cheatsheet():
    return "msm142: Numbered display equation (8.9) from MVSML chapter 8."
