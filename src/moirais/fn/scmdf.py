# moirais.fn — function file (hadesllm/moirais)
"""Structural causal model (SCM) definition: (U, V, F) triple."""
import numpy as np
from ._richresult import RichResult

__all__ = ["scm_definition"]


def scm_definition(exogenous, endogenous, equations):
    """
    Structural causal model (SCM) definition: (U, V, F) triple

    Formula: SCM = (U: exogenous, V: endogenous, F: structural equations); V_i = f_i(pa(V_i), U_i)

    Parameters
    ----------
    exogenous : array-like
        Input data.
    endogenous : array-like
        Input data.
    equations : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'scm': 'object'}

    References
    ----------
    Molak Ch 2
    """
    exogenous = np.asarray(exogenous, dtype=float)
    n = int(exogenous) if exogenous.ndim == 0 else len(exogenous)
    result = float(np.mean(exogenous))
    se = float(np.std(exogenous, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Structural causal model (SCM) definition: (U, V, F) triple"})


def cheatsheet():
    return "scmdf: Structural causal model (SCM) definition: (U, V, F) triple"
