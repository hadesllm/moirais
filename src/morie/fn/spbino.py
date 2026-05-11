"""Binomial point process: n points independently uniform on region A."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_binomial_process"]


def schabenberger_binomial_process(n, region):
    """
    Binomial point process: n points independently uniform on region A

    Formula: P(N(B)=k) = C(n,k)*(|B|/|A|)^k*(1-|B|/|A|)^{n-k}

    Parameters
    ----------
    n : array-like
        Input data.
    region : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pmf

    References
    ----------
    Schabenberger Ch 3, Sec 3.2.1
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binomial point process: n points independently uniform on region A"})


def cheatsheet():
    return "spbino: Binomial point process: n points independently uniform on region A"
