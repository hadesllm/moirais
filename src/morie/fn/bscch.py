# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Binary symmetric channel capacity."""

__all__ = ["bscch"]

import numpy as np
from ._richresult import RichResult


def bscch(p: float) -> dict:
    """
    Capacity of a binary symmetric channel (BSC).

    .. math::

        C = 1 - H(p) = 1 + p \\log_2 p + (1-p) \\log_2 (1-p)

    Parameters
    ----------
    p : float
        Crossover probability, 0 <= p <= 1.

    Returns
    -------
    dict
        'capacity' (bits), 'crossover_prob'.

    Raises
    ------
    ValueError
        If p not in [0, 1].

    References
    ----------
    Cover & Thomas (2006). Elements of Information Theory, Ch. 7.
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    if p == 0 or p == 1:
        h = 0.0
    else:
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return RichResult(payload={"capacity": 1.0 - h, "crossover_prob": p})
