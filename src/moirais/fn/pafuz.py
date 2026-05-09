# moirais.fn — function file (hadesllm/moirais)
"""Fuzzy preference aggregation."""

import numpy as np

from ._containers import SpatialResult


def pafuz(ideal, pos, fuzz=1.0):
    """Fuzzy preference aggregation.

    Parameters
    ----------
    ideal : array_like
        Ideal point coordinates.
    pos : array_like
        Position coordinates.
    fuzz : float
        Fuzziness parameter (default 1.0).

    Returns
    -------
    SpatialResult
    """
    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    d = float(np.linalg.norm(ideal - pos))
    membership = float(np.exp(-(d**2) / (2 * fuzz**2)))
    stat = membership
    return SpatialResult(
        name="Fuzzy Membership Utility",
        statistic=float(stat),
        extra={"distance": d, "membership": membership},
    )


pafuz = pafuz  # alias


def cheatsheet() -> str:
    return "pafuz({}) -> Fuzzy preference aggregation."
