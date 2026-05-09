# moirais.fn — function file (hadesllm/moirais)
"""Distance from agreement. 'Special Beam Cannon!' -- Piccolo, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def distance_from_agreement(agreement):
    """Convert agreement matrix to distance matrix: D = 1 - agreement.

    Parameters
    ----------
    agreement : array-like
        Agreement matrix with values in [0, 1].

    Returns
    -------
    DescriptiveResult
        value = distance matrix.
    """
    import numpy as np

    A = np.asarray(agreement, dtype=float)
    D = 1.0 - A
    return DescriptiveResult(name="distance_from_agreement", value=D, extra={"shape": D.shape})


rcdst = distance_from_agreement


def cheatsheet() -> str:
    return "distance_from_agreement({}) -> Distance from agreement. 'Special Beam Cannon!' -- Piccolo, "
