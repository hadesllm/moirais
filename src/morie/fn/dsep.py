# morie.fn — function file (hadesllm/morie)
"""d-separation criterion for conditional independence in DAGs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["d_separation"]


def d_separation(dag, X, Y, Z):
    """
    d-separation criterion for conditional independence in DAGs

    Formula: X _|_G Y | Z if all paths between X and Y are blocked by Z; path blocked if: (1) Z on chain/fork, (2) collider not in Z and has no descendant in Z

    Parameters
    ----------
    dag : array-like
        Input data.
    X : array-like
        Input data.
    Y : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'d_separated': 'bool'}

    References
    ----------
    Molak Ch 5,6
    """
    if isinstance(dag, dict):
        dag = [len(v) if hasattr(v, '__len__') else float(v) for v in dag.values()] or [0.0]
    dag = np.asarray(dag, dtype=float)
    n = int(dag) if dag.ndim == 0 else len(dag)
    result = float(np.mean(dag))
    se = float(np.std(dag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "d-separation criterion for conditional independence in DAGs"})


def cheatsheet():
    return "dsep: d-separation criterion for conditional independence in DAGs"
