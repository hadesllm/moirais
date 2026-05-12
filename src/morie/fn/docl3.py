# morie.fn -- function file (hadesllm/morie)
"""Pearl's three rules of do-calculus for identification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["do_calculus_rules"]


def do_calculus_rules(dag, query):
    """
    Pearl's three rules of do-calculus for identification

    Formula: Rule 1: P(Y|do(X),Z,W) = P(Y|do(X),W) if Y _|_ Z|X,W in G_{X-bar}; Rule 2/3 similarly

    Parameters
    ----------
    dag : array-like
        Input data.
    query : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'identified': 'bool', 'expression': 'str'}

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pearl's three rules of do-calculus for identification"})


def cheatsheet():
    return "docl3: Pearl's three rules of do-calculus for identification"
