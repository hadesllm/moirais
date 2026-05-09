# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""B-spline wavelet construction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "This is a new day, a new beginning."


def _bspline(t, order):
    if order == 0:
        return np.where((t >= -0.5) & (t < 0.5), 1.0, 0.0)
    prev = _bspline(t, order - 1)
    result = np.convolve(prev, np.ones(len(t)) / len(t), mode="same")
    return result


def bspline_wavelet(order: int = 3, scale: float = 1.0) -> DescriptiveResult:
    """Construct a B-spline wavelet.

    Parameters
    ----------
    order : int
        Spline order (default 3 = cubic).
    scale : float
        Scale parameter (default 1.0).

    Returns
    -------
    DescriptiveResult
    """
    n_points = max(64, int(8 * scale * (order + 1)))
    t = np.linspace(-order - 1, order + 1, n_points) / scale

    b = np.where((t >= -0.5) & (t < 0.5), 1.0, 0.0)
    box = np.ones(n_points) / n_points
    for _ in range(order):
        b = np.convolve(b, box, mode="same")

    psi = np.diff(np.concatenate([[0], b]))
    psi = psi / (np.sqrt(np.sum(psi**2)) + 1e-12)

    return DescriptiveResult(
        name="bspline_wavelet",
        value=float(order),
        extra={"wavelet": psi, "t": t[: len(psi)], "order": order, "scale": scale},
    )


bspwv = bspline_wavelet


def cheatsheet() -> str:
    return "_bspline({}) -> B-spline wavelet construction."
