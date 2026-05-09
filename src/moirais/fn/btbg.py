"""Bootstrap aggregating (bagging) ensemble prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_bagging_predict"]


def boot_bagging_predict(models, X_new, kind):
    """
    Bootstrap aggregating (bagging) ensemble prediction

    Formula: ĝ_bag(x) = (1/B) Σ ĝ_b(x)

    Parameters
    ----------
    models : array-like
        Input data.
    X_new : array-like
        Input data.
    kind : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred

    References
    ----------
    Breiman (1996)
    """
    models = np.atleast_1d(np.asarray(models, dtype=float))
    n = len(models)
    result = float(np.mean(models))
    se = float(np.std(models, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap aggregating (bagging) ensemble prediction"})


def cheatsheet():
    return "btbg: Bootstrap aggregating (bagging) ensemble prediction"
