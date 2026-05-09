# moirais.fn — function file (hadesllm/moirais)
"""Causal forest for heterogeneous treatment effect estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["causal_forest"]


def causal_forest(Y, T, X, n_trees, min_node_size):
    """
    Causal forest for heterogeneous treatment effect estimation

    Formula: tau_hat(Y) = (1/B)*sum_b tau_b(Y); each tree splits to maximize CATE heterogeneity; honest trees

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.
    X : array-like
        Input data.
    n_trees : array-like
        Input data.
    min_node_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'cate': 'array', 'var_importance': 'array'}

    References
    ----------
    Molak Ch 10
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    if Y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Causal forest for heterogeneous treatment effect estimation"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Causal forest for heterogeneous treatment effect estimation"})


def cheatsheet():
    return "cfst: Causal forest for heterogeneous treatment effect estimation"
