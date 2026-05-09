# moirais.fn — function file (hadesllm/moirais)
"""Distance to squared element-wise. 'Believe it!' -- Naruto, Naruto"""

from __future__ import annotations

from ._containers import DescriptiveResult


def distance_to_squared(D):
    """Square each element of distance matrix D.

    Parameters
    ----------
    D : array-like
        Square distance matrix.

    Returns
    -------
    DescriptiveResult
        value = D_squared (ndarray), extra has shape.
    """
    import numpy as np

    D = np.asarray(D, dtype=float)
    D2 = D**2
    return DescriptiveResult(name="distance_to_squared", value=D2, extra={"shape": D2.shape})


dst2s = distance_to_squared


def cheatsheet() -> str:
    return "distance_to_squared({}) -> Distance to squared element-wise. 'Believe it!' -- Naruto, N"
