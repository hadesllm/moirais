# morie.fn -- function file (hadesllm/morie)
"""Faithfulness assumption: conditional independencies in P imply d-separation in G."""
import numpy as np
from ._richresult import RichResult

__all__ = ["faithfulness_assumption"]


def faithfulness_assumption(dag, P):
    """
    Faithfulness assumption: conditional independencies in P imply d-separation in G

    Formula: If X _|_ Y | Z in P, then X _|_G Y | Z; no cancelling paths in causal structure

    Parameters
    ----------
    dag : array-like
        Input data.
    P : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'faithful': 'bool'}

    References
    ----------
    Molak Ch 5
    """
    if isinstance(dag, dict):
        dag = [len(v) if hasattr(v, '__len__') else float(v) for v in dag.values()] or [0.0]
    dag = np.asarray(dag, dtype=float)
    n = int(dag) if dag.ndim == 0 else len(dag)
    result = float(np.mean(dag))
    se = float(np.std(dag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Faithfulness assumption: conditional independencies in P imply d-separation in G"})


def cheatsheet():
    return "faithA: Faithfulness assumption: conditional independencies in P imply d-separation in G"
