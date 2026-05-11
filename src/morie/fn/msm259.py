"""Numbered display equation (13.1) from MVSML chapter 13.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_deep_learning_eq_13_1"]


def mvsml_deep_learning_eq_13_1(image, of, size, Part, b, Fig):
    """
    Numbered display equation (13.1) from MVSML chapter 13.

    Formula: image of size 256  256  3. Part (b) of Fig. 13.16 exempliﬁes the ﬁlter matching operation using CNNs, and we can see clearly that the matching process is done locally, that is, in small patches or overlapping patches of an image. Filter matching is done by computing the pre-activation (zi) and activation (yi) values with the following equations. This is done at each position of the ﬁlter. X 147 zi = w jxij + b

    Parameters
    ----------
    image : array-like
        Input data.
    of : array-like
        Input data.
    size : array-like
        Input data.
    Part : array-like
        Input data.
    b : array-like
        Input data.
    Fig : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (13.1) [Multivariate Statistical Machine Learnin [Pages 533-577] [2026-04-16].pdf]
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (13.1) from MVSML chapter 13."})


def cheatsheet():
    return "msm259: Numbered display equation (13.1) from MVSML chapter 13."
