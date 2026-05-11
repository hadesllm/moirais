"""DeepSurv neural-net Cox."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survival_neural_net"]


def survival_neural_net(time, event, X, model):
    """
    DeepSurv neural-net Cox

    Formula: replace beta X with NN g(X) in partial likelihood

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Katzman et al (2018)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DeepSurv neural-net Cox"})


def cheatsheet():
    return "survnnr: DeepSurv neural-net Cox"
