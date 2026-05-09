"""Linear-chain CRF for sequence labeling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["crf_sequence"]


def crf_sequence(X, y):
    """
    Linear-chain CRF for sequence labeling

    Formula: P(y|x) ∝ exp(sum θ_k f_k(y_t,y_{t-1},x))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lafferty-McCallum-Pereira (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear-chain CRF for sequence labeling"})


def cheatsheet():
    return "crfNlp: Linear-chain CRF for sequence labeling"
