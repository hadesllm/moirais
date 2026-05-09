# moirais.fn — function file (hadesllm/moirais)
"""Chebyshev distance."""

import numpy as np

from ._containers import ESRes
def chebyshev_dist(x, y, **kwargs) -> ESRes:
    """
    Compute Chebyshev (L-infinity) distance.

    .. math::

        d(x, y) = \\max_i |x_i - y_i|

    :param x: array-like, first vector.
    :param y: array-like, second vector.
    :return: ESRes with Chebyshev distance.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    d = float(np.max(np.abs(x - y)))
    idx = int(np.argmax(np.abs(x - y)))
    return ESRes(measure="chebyshev_distance", estimate=d, extra={"argmax_dim": idx, "dim": len(x)})


chbyd = chebyshev_dist


def cheatsheet() -> str:
    return "chebyshev_dist({}) -> Chebyshev distance."
