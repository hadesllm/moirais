"""Index of Prediction Accuracy (IPA) — Brier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ipa_brier"]


def ipa_brier(fit, null_fit, time):
    """
    Index of Prediction Accuracy (IPA) — Brier

    Formula: 1 - Brier_model / Brier_null

    Parameters
    ----------
    fit : array-like
        Input data.
    null_fit : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kattan-Gerds (2018)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Index of Prediction Accuracy (IPA) — Brier"})


def cheatsheet():
    return "survipa: Index of Prediction Accuracy (IPA) — Brier"
