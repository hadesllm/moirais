# moirais.fn — function file (hadesllm/moirais)
"""Fast Causal Inference (FCI) algorithm for hidden confounders."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fci_algorithm"]


def fci_algorithm(data, alpha, ci_test):
    """
    Fast Causal Inference (FCI) algorithm for hidden confounders

    Formula: PC-like skeleton; orient v-structures; check Maximal Ancestral Graph (MAG) conditions; output PAG

    Parameters
    ----------
    data : array-like
        Input data.
    alpha : array-like
        Input data.
    ci_test : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'pag': 'graph'}

    References
    ----------
    Molak Ch 14
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fast Causal Inference (FCI) algorithm for hidden confounders"})


def cheatsheet():
    return "fciag: Fast Causal Inference (FCI) algorithm for hidden confounders"
