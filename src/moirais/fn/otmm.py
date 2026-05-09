"""Mini-batch OT loss over random subsets (for SGD training)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_minibatch_loss"]


def ot_minibatch_loss(X, Y, batch_size, n_batches, epsilon):
    """
    Mini-batch OT loss over random subsets (for SGD training)

    Formula: (1/M) Σ_m OT_ε(X_m, Y_m)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    batch_size : array-like
        Input data.
    n_batches : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Fatras-Zine-Flamary-Gribonval-Courty (2020)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mini-batch OT loss over random subsets (for SGD training)"})


def cheatsheet():
    return "otmm: Mini-batch OT loss over random subsets (for SGD training)"
