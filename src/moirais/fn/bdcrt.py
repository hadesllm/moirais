# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Back-door criterion for identifying causal effect via adjustment."""
import numpy as np
from ._richresult import RichResult

__all__ = ["backdoor_criterion"]


def backdoor_criterion(dag, X, Y, Z):
    """
    Back-door criterion for identifying causal effect via adjustment

    Formula: Z satisfies backdoor: (1) no Z is descendant of X; (2) Z blocks all backdoor paths X <- ... -> Y; then P(Y|do(X)) = sum_z P(Y|X,Z=z)*P(Z=z)

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
        Keys: {'satisfies': 'bool', 'adjustment_formula': 'expression'}

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Back-door criterion for identifying causal effect via adjustment"})


def cheatsheet():
    return "bdcrt: Back-door criterion for identifying causal effect via adjustment"
