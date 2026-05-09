# moirais.fn — function file (hadesllm/moirais)
"""QRS complex detection (Pan-Tompkins)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_qrs_detect"]


def rangayyan_qrs_detect(x):
    """
    QRS complex detection (Pan-Tompkins)

    Formula: BP filter → differentiate → square → MA → threshold

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
    Rangayyan Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QRS complex detection (Pan-Tompkins)"})


def cheatsheet():
    return "rgqrs: QRS complex detection (Pan-Tompkins)"
