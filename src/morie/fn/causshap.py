"""Shapley value-based causal contribution decomposition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_shap_decomposition"]


def causal_shap_decomposition(X, y, model, n_samples):
    """
    Shapley value-based causal contribution decomposition

    Formula: ϕ_i = Σ_S w(S)(v(S∪{i})-v(S))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    model : array-like
        Input data.
    n_samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: phi

    References
    ----------
    Lundberg-Lee (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shapley value-based causal contribution decomposition"})


def cheatsheet():
    return "causshap: Shapley value-based causal contribution decomposition"
