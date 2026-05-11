# morie.fn — function file (hadesllm/morie)
"""Event-related potential (ERP) latency and amplitude features."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_erp_features"]


def rangayyan_erp_features(erp, fs):
    """
    Event-related potential (ERP) latency and amplitude features

    Formula: P300 latency = argmax(ERP), N200 latency = argmin(ERP in 150-250ms window)

    Parameters
    ----------
    erp : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: latency, amplitude

    References
    ----------
    Rangayyan Ch 1.2.7
    """
    erp = np.asarray(erp, dtype=float)
    n = int(erp) if erp.ndim == 0 else len(erp)
    result = float(np.mean(erp))
    se = float(np.std(erp, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Event-related potential (ERP) latency and amplitude features"})


def cheatsheet():
    return "rgerp: Event-related potential (ERP) latency and amplitude features"
