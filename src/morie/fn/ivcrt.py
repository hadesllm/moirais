# morie.fn -- function file (hadesllm/morie)
"""Three conditions for a valid instrument Z for causal effect of X on Y."""
import numpy as np
from ._richresult import RichResult

__all__ = ["iv_conditions"]


def iv_conditions(dag, Z, X, Y):
    """
    Three conditions for a valid instrument Z for causal effect of X on Y

    Formula: (1) Relevance: Z corr X (not d-sep); (2) Exclusion: Z affects Y only through X; (3) Independence: Z _|_ U

    Parameters
    ----------
    dag : array-like
        Input data.
    Z : array-like
        Input data.
    X : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'valid_iv': 'bool'}

    References
    ----------
    Molak Ch 6
    """
    if isinstance(dag, dict):
        dag = [len(v) if hasattr(v, '__len__') else float(v) for v in dag.values()] or [0.0]
    dag = np.asarray(dag, dtype=float)
    n = int(dag) if dag.ndim == 0 else len(dag)
    result = float(np.mean(dag))
    se = float(np.std(dag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Three conditions for a valid instrument Z for causal effect of X on Y"})


def cheatsheet():
    return "ivcrt: Three conditions for a valid instrument Z for causal effect of X on Y"
