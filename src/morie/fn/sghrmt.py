"""Hermite polynomial evaluation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def hermite_polynomial(x: np.ndarray, n: int = 3) -> SpatialResult:
    r"""Evaluate the probabilist's Hermite polynomial of degree *n*.

    .. math::

        He_n(x) = x\,He_{n-1}(x) - (n-1)\,He_{n-2}(x)

    Parameters
    ----------
    x : np.ndarray
        Evaluation points.
    n : int
        Degree of Hermite polynomial.

    Returns
    -------
    SpatialResult
        ``statistic`` is the mean of :math:`He_n(x)`.
        ``extra`` has ``values``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "Wake me when you need me." -- Master Chief, Halo
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        vals = np.ones_like(x)
    elif n == 1:
        vals = x.copy()
    else:
        h_prev = np.ones_like(x)
        h_curr = x.copy()
        for k in range(2, n + 1):
            h_next = x * h_curr - (k - 1) * h_prev
            h_prev, h_curr = h_curr, h_next
        vals = h_curr

    return SpatialResult(
        name="hermite_polynomial",
        statistic=float(np.mean(vals)),
        p_value=None,
        extra={"values": vals, "degree": n},
    )


sghrmt = hermite_polynomial


def cheatsheet() -> str:
    return "hermite_polynomial({}) -> Hermite polynomial evaluation."
