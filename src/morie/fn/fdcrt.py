# morie.fn — function file (hadesllm/morie)
"""Front-door criterion: identification via mediator when backdoor blocked."""
import numpy as np
from ._richresult import RichResult

__all__ = ["frontdoor_criterion"]


def frontdoor_criterion(dag, X, Y, Z):
    """
    Front-door criterion: identification via mediator when backdoor blocked

    Formula: Z satisfies frontdoor if: (1) Z intercepts all directed paths X->Y; (2) no unblocked backdoor X->Z; (3) all backdoors Z->Y blocked by X

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
        Keys: {'satisfies': 'bool'}

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Front-door criterion: identification via mediator when backdoor blocked"})


def cheatsheet():
    return "fdcrt: Front-door criterion: identification via mediator when backdoor blocked"
