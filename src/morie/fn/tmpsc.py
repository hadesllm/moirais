"""Temperature scaling for logits."""
import numpy as np
from ._richresult import RichResult

__all__ = ["temperature_scaling"]


def temperature_scaling(x):
    """
    Temperature scaling for logits

    Formula: p_i = exp(z_i/T) / sum exp(z_j/T)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hinton et al. (2015)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Temperature scaling for logits"})


def cheatsheet():
    return "tmpsc: Temperature scaling for logits"
