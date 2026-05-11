"""Numbered display equation (6.8) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_8"]


def mvsml_bayesian_regression_eq_6_8(An, of, this, model, can, be):
    """
    Numbered display equation (6.8) from MVSML chapter 6.

    Formula: An implementation of this model can be done using the github version of the BGLR R library, which can be accessed from https://github.com/gdlc/BGLR-R and can be installed directly in the R console by running the following commands: install.packages('devtools'); library(devtools); install_git('https://github.com/gdlc/ BGLR-R'). This implementation also uses a ﬂat prior for the ﬁxed effect regression coefﬁcients \beta, and in such a case, the corresponding full conditional of this parameter is the same as step 1 of the Gibbs sampler given before, but removing \Sigma-1 \beta and \Sigma-1 \beta \beta0 from e\Sigma\beta and e\beta0, respectively. Speciﬁcally, model

    Parameters
    ----------
    An : array-like
        Input data.
    of : array-like
        Input data.
    this : array-like
        Input data.
    model : array-like
        Input data.
    can : array-like
        Input data.
    be : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.8) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    An = np.atleast_1d(np.asarray(An, dtype=float))
    n = len(An)
    result = float(np.mean(An))
    se = float(np.std(An, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.8) from MVSML chapter 6."})


def cheatsheet():
    return "msm069: Numbered display equation (6.8) from MVSML chapter 6."
