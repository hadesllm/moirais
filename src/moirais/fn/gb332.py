# moirais.fn — function file (hadesllm/moirais)
"""Distribution of lengths of runs of type 1 objects marginally."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_type1_run_lengths"]


def gibbons_type1_run_lengths(run_lengths, n1, n2):
    """
    Distribution of lengths of runs of type 1 objects marginally

    Formula: f(r11,...,r1n1) = r1!*C(n2+1,r1) / (prod(r1j!) * C(n1+n2,n1))

    Parameters
    ----------
    run_lengths : array-like
        Input data.
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Theorem 3.3.2
    """
    run_lengths = np.asarray(run_lengths, dtype=float)
    n = int(run_lengths) if run_lengths.ndim == 0 else len(run_lengths)
    result = float(np.mean(run_lengths))
    se = float(np.std(run_lengths, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Distribution of lengths of runs of type 1 objects marginally"})


def cheatsheet():
    return "gb332: Distribution of lengths of runs of type 1 objects marginally"
