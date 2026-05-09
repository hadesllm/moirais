# moirais.fn — function file (hadesllm/moirais)
"""EMG root mean square envelope."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_emg_rms"]


def rangayyan_emg_rms(x):
    """
    EMG root mean square envelope

    Formula: RMS = sqrt((1/W) sum x[n]^2)

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
    Rangayyan Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EMG root mean square envelope"})


def cheatsheet():
    return "rgemg: EMG root mean square envelope"
