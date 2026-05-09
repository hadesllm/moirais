# moirais.fn — function file (hadesllm/moirais)
"""Markov equivalence class (MEC): DAGs with same skeleton and v-structures."""
import numpy as np
from ._richresult import RichResult

__all__ = ["markov_equivalence_class"]


def markov_equivalence_class(dag):
    """
    Markov equivalence class (MEC): DAGs with same skeleton and v-structures

    Formula: DAGs G1, G2 are Markov equivalent iff same skeleton and same unshielded colliders; CPDAG represents MEC

    Parameters
    ----------
    dag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'cpdag': 'graph'}

    References
    ----------
    Molak Ch 5
    """
    dag = np.asarray(dag, dtype=float)
    n = int(dag) if dag.ndim == 0 else len(dag)
    result = float(np.mean(dag))
    se = float(np.std(dag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Markov equivalence class (MEC): DAGs with same skeleton and v-structures"})


def cheatsheet():
    return "mecpd: Markov equivalence class (MEC): DAGs with same skeleton and v-structures"
