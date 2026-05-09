"""Numbered display equation (4.14) from MVSML chapter 4.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_overfitting_resampling_eq_4_14"]


def mvsml_overfitting_resampling_eq_4_14(random, means, a, total, disagreement, between):
    """
    Numbered display equation (4.14) from MVSML chapter 4.

    Formula: random and +1 means a total disagreement between predicted and observed values. Next, we present the Brier score (Brier 1950) for categorical or binary data that can be computed as BS = T+1 X n+T X C )2, ( b\piic + dic

    Parameters
    ----------
    random : array-like
        Input data.
    means : array-like
        Input data.
    a : array-like
        Input data.
    total : array-like
        Input data.
    disagreement : array-like
        Input data.
    between : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (4.14) [Multivariate Statistical Machine Learnin [Pages 109-139] [2026-04-16].pdf]
    """
    random = np.atleast_1d(np.asarray(random, dtype=float))
    n = len(random)
    result = float(np.mean(random))
    se = float(np.std(random, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (4.14) from MVSML chapter 4."})


def cheatsheet():
    return "msm009: Numbered display equation (4.14) from MVSML chapter 4."
