"""Series adapter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_series_adapter"]


def kamath_ch4_series_adapter(H_o, W_down, W_up):
    """
    Series adapter.

    Formula: H_o \leftarrow H_o + f(H_o W_{down}) W_{up}

    Parameters
    ----------
    H_o : array-like
        Input data.
    W_down : array-like
        Input data.
    W_up : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 4, Eq 4.1, p. 147
    """
    H_o = np.atleast_1d(np.asarray(H_o, dtype=float))
    n = len(H_o)
    result = float(np.mean(H_o))
    se = float(np.std(H_o, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Series adapter."})


def cheatsheet():
    return "km054: Series adapter."
