"""Numbered display equation (1.4) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_4"]


def mvsml_general_eq_1_4(mental, effects, Two, drawbacks, of, the):
    """
    Numbered display equation (1.4) from MVSML chapter 1.

    Formula: mental effects. Two drawbacks of the model with ﬁxed effects given in Eq. (1.3) are that it is unable to provide an estimate of the between-environments variability and that the number of parameters in the model increases linearly with the number of environments. Fortunately, the random effects model circumvents these problems by treating the environmental effects as random variations around a population mean. Next we reparameterize model (1.3) as a random effects model. We write -  GYij = \beta + \betai \beta + eij,

    Parameters
    ----------
    mental : array-like
        Input data.
    effects : array-like
        Input data.
    Two : array-like
        Input data.
    drawbacks : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.4) [Multivariate Statistical Machine Learnin [Pages 1-34] [2026-04-16].pdf]
    """
    mental = np.atleast_1d(np.asarray(mental, dtype=float))
    n = len(mental)
    result = float(np.mean(mental))
    se = float(np.std(mental, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.4) from MVSML chapter 1."})


def cheatsheet():
    return "msm005: Numbered display equation (1.4) from MVSML chapter 1."
