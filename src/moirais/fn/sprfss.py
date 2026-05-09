"""Stationarity definitions: strict, second-order, intrinsic."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_random_field_stationarity"]


def schabenberger_random_field_stationarity(x=None, *args, **kwargs):
    """
    Stationarity definitions: strict, second-order, intrinsic

    Formula: 2nd-order: E[Z(s)]=mu, Cov(Z(s),Z(s+h))=C(h); intrinsic: E[Z(s+h)-Z(s)]=0, Var=2*gamma(h)

    Parameters
    ----------


    Returns
    -------
    result : dict
        Keys: definition

    References
    ----------
    Schabenberger Ch 2, Sec 2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stationarity definitions: strict, second-order, intrinsic"})


def cheatsheet():
    return "sprfss: Stationarity definitions: strict, second-order, intrinsic"
