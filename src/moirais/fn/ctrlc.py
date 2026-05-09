# moirais.fn — function file (hadesllm/moirais)
"""Comparisons with a control (Dunnett-type nonparametric)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["control_comparison"]


def control_comparison(x):
    """
    Comparisons with a control (Dunnett-type nonparametric)

    Formula: Multiple comparisons against control group

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Gibbons Ch 10.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Comparisons with a control (Dunnett-type nonparametric)"})


def cheatsheet():
    return "ctrlc: Comparisons with a control (Dunnett-type nonparametric)"
