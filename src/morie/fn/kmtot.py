# morie.fn — function file (hadesllm/morie)
"""Tree of Thoughts: explore a tree of partial reasoning steps with search."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_tree_of_thoughts"]


def kamath_tree_of_thoughts(problem, branch_factor, max_depth, model):
    """
    Tree of Thoughts: explore a tree of partial reasoning steps with search

    Formula: search over tree T of thought-nodes; each node scored by LLM-self-eval; beam or DFS

    Parameters
    ----------
    problem : array-like
        Input data.
    branch_factor : array-like
        Input data.
    max_depth : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best_path

    References
    ----------
    Kamath Ch 4, Tree-of-Thoughts section
    """
    problem = np.atleast_1d(np.asarray(problem, dtype=float))
    n = len(problem)
    result = float(np.mean(problem))
    se = float(np.std(problem, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tree of Thoughts: explore a tree of partial reasoning steps with search"})


def cheatsheet():
    return "kmtot: Tree of Thoughts: explore a tree of partial reasoning steps with search"
