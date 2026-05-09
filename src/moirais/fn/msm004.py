"""Numbered display equation (1.3) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_3"]


def mvsml_general_eq_1_3(re, centered, around, zero, than, the):
    """
    Numbered display equation (1.3) from MVSML chapter 1.

    Formula: re centered around zero than in the ﬁrst ﬁtted model (single-mean model), which can be observed in the residual standard error that is 7.47 times smaller. Therefore, we have evidence that the model given in Eq. (1.3) successfully accounted for the environ- mental effects. Two drawbacks of the model with ﬁxed effects given in Eq. (1.3) are that it is unable to provide an estimate of the between-environments variability and that the number of parameters in the model increases linearly with the number of environments. Fortunately, the random effects model circumvents these problems by treating the environmental effects as random variations around a population mean. Next we reparameterize model

    Parameters
    ----------
    re : array-like
        Input data.
    centered : array-like
        Input data.
    around : array-like
        Input data.
    zero : array-like
        Input data.
    than : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.3) [Multivariate Statistical Machine Learnin [Pages 1-34] [2026-04-16].pdf]
    """
    re = np.atleast_1d(np.asarray(re, dtype=float))
    n = len(re)
    result = float(np.mean(re))
    se = float(np.std(re, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.3) from MVSML chapter 1."})


def cheatsheet():
    return "msm004: Numbered display equation (1.3) from MVSML chapter 1."
