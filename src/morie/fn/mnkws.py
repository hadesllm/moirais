# morie.fn — function file (hadesllm/morie)
"""Minkowski metric and spacetime interval computation."""

__all__ = ["mnkws"]

import numpy as np


def mnkws(
    event_a: np.ndarray,
    event_b: np.ndarray,
    signature: str = "mostly_minus",
) -> dict:
    """
    Compute the Minkowski spacetime interval between two events.

    For signature (+,-,-,-) (mostly_minus):

    .. math::

        s^2 = c^2 \\Delta t^2 - \\Delta x^2 - \\Delta y^2 - \\Delta z^2

    Parameters
    ----------
    event_a, event_b : np.ndarray
        4-vectors [ct, x, y, z].
    signature : str
        "mostly_minus" for (+,-,-,-) or "mostly_plus" for (-,+,+,+).

    Returns
    -------
    dict
        Keys: interval_squared, interval_type, metric, separation.
    """
    a = np.asarray(event_a, dtype=float)
    b = np.asarray(event_b, dtype=float)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("Events must be length-4 arrays [ct, x, y, z].")

    if signature == "mostly_minus":
        eta = np.diag([1.0, -1.0, -1.0, -1.0])
    elif signature == "mostly_plus":
        eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    else:
        raise ValueError("signature must be 'mostly_minus' or 'mostly_plus'.")

    dx = b - a
    s2 = float(dx @ eta @ dx)

    if abs(s2) < 1e-12:
        itype = "lightlike"
    elif s2 > 0:
        itype = "timelike" if signature == "mostly_minus" else "spacelike"
    else:
        itype = "spacelike" if signature == "mostly_minus" else "timelike"

    return {
        "interval_squared": s2,
        "interval_type": itype,
        "metric": eta,
        "separation": dx,
    }
