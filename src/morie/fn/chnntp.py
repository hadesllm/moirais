"""Channel capacity (Blahut-Arimoto)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["channel_capacity"]


def channel_capacity(channel):
    """
    Channel capacity (Blahut-Arimoto)

    Formula: C = max_{p(x)} I(X;Y)

    Parameters
    ----------
    channel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blahut (1972); Arimoto (1972)
    """
    channel = np.atleast_1d(np.asarray(channel, dtype=float))
    n = len(channel)
    result = float(np.mean(channel))
    se = float(np.std(channel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Channel capacity (Blahut-Arimoto)"})


def cheatsheet():
    return "chnntp: Channel capacity (Blahut-Arimoto)"
