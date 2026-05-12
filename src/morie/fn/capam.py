# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Truth comes out of error more readily than out of confusion. — Francis Bacon"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def circular_mean(
    angles: np.ndarray | list[float],
    *,
    degrees: bool = True,
    weights: np.ndarray | None = None,
) -> DescriptiveResult:
    r"""Compute circular (directional) mean and concentration parameter.

    Parameters
    ----------
    angles : array-like
        Angular observations.
    degrees : bool
        If True, input is in degrees (converted internally to radians).
    weights : array-like or None
        Optional weights for each observation.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``mean_direction`` (in input units),
        ``mean_resultant_length`` :math:`\\bar{R}`, ``concentration``
        (von Mises :math:`\\hat{\\kappa}` estimate), ``n``.
    """
    a = np.asarray(angles, dtype=float)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("angles must be a non-empty 1D array")

    if degrees:
        theta = np.deg2rad(a)
    else:
        theta = a.copy()

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(theta):
            raise ValueError("weights must match angles length")
        w = w / w.sum()
    else:
        w = np.ones(len(theta)) / len(theta)

    C = np.sum(w * np.cos(theta))
    S = np.sum(w * np.sin(theta))
    mean_dir = np.arctan2(S, C)
    R_bar = np.sqrt(C**2 + S**2)

    if R_bar < 1e-10:
        kappa = 0.0
    elif R_bar < 0.53:
        kappa = 2 * R_bar + R_bar**3 + 5 * R_bar**5 / 6
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (1 - R_bar)
    else:
        kappa = 1.0 / (R_bar**3 - 4 * R_bar**2 + 3 * R_bar) if R_bar < 1.0 else 1e6

    if degrees:
        mean_dir_out = float(np.rad2deg(mean_dir)) % 360
    else:
        mean_dir_out = float(mean_dir) % (2 * np.pi)

    return DescriptiveResult(
        name="circular_mean",
        value={
            "mean_direction": mean_dir_out,
            "mean_resultant_length": float(R_bar),
            "concentration": float(kappa),
            "n": len(a),
        },
        extra={"degrees": degrees},
    )


capam = circular_mean


def cheatsheet() -> str:
    return "circular_mean({}) -> Circular statistics — mean direction and concentration. 'I c"
