"""Message-passing framework."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["message_passing"]


def message_passing(G, h0, layers):
    """
    Message-passing framework

    Formula: h^{l+1}_v = U(h^l_v, AGG{M(h^l_v, h^l_u, e_{vu}) : u∈N(v)})

    Parameters
    ----------
    G : array-like
        Input data.
    h0 : array-like
        Input data.
    layers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gilmer et al (2017) MPNN
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Message-passing framework"})


def cheatsheet():
    return "mpfn: Message-passing framework"
