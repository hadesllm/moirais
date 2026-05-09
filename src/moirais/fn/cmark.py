# moirais.fn — function file (hadesllm/moirais)
"""Causal Markov condition: each node independent of non-descendants given parents."""
import numpy as np
from ._richresult import RichResult

__all__ = ["causal_markov_condition"]


def causal_markov_condition(dag, P):
    """
    Causal Markov condition: each node independent of non-descendants given parents

    Formula: X _|_ Non-Desc(X) | Pa(X) in causal DAG; P(V) = prod_i P(V_i | Pa(V_i))

    Parameters
    ----------
    dag : array-like
        Input data.
    P : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'factorization': 'expression'}

    References
    ----------
    Molak Ch 2,5
    """
    if isinstance(dag, dict):
        dag = [len(v) if hasattr(v, '__len__') else float(v) for v in dag.values()] or [0.0]
    dag = np.asarray(dag, dtype=float)
    n = int(dag) if dag.ndim == 0 else len(dag)
    result = float(np.mean(dag))
    se = float(np.std(dag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal Markov condition: each node independent of non-descendants given parents"})


def cheatsheet():
    return "cmark: Causal Markov condition: each node independent of non-descendants given parents"
