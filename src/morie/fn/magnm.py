# morie.fn — function file (hadesllm/morie)
"""Biot-Savart magnetic field computation. 'You are all beneath me.' -- Magneto"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def biot_savart(
    wire_points: np.ndarray,
    current: float,
    field_point: np.ndarray,
    *,
    mu0: float = 4e-7 * np.pi,
) -> DescriptiveResult:
    r"""Compute the magnetic field at a point due to a current-carrying wire
    using the Biot-Savart law.

    :math:`\\mathbf{B} = \\frac{\\mu_0 I}{4\\pi} \\int \\frac{d\\mathbf{l} \\times \\hat{r}}{r^2}`

    The wire is discretised as a polyline; each segment contributes via the
    finite-length Biot-Savart formula.

    Parameters
    ----------
    wire_points : np.ndarray
        (M x 3) array of wire segment endpoints in metres.
    current : float
        Current in amperes.
    field_point : np.ndarray
        (3,) point at which to evaluate B.
    mu0 : float
        Permeability of free space (default: vacuum).

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``B`` (3-vector in Tesla), ``B_mag`` (scalar).
    """
    pts = np.asarray(wire_points, dtype=float)
    fp = np.asarray(field_point, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("wire_points must be (M x 3)")
    if fp.shape != (3,):
        raise ValueError("field_point must be (3,)")
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 wire points")

    B = np.zeros(3)
    for i in range(len(pts) - 1):
        dl = pts[i + 1] - pts[i]
        midpoint = (pts[i] + pts[i + 1]) / 2
        r_vec = fp - midpoint
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-15:
            continue
        dB = (mu0 * current / (4 * np.pi)) * np.cross(dl, r_vec) / r_mag**3
        B += dB

    return DescriptiveResult(
        name="biot_savart",
        value={"B": B, "B_mag": float(np.linalg.norm(B))},
        extra={"current": current, "n_segments": len(pts) - 1},
    )


magnm = biot_savart


def cheatsheet() -> str:
    return "biot_savart({}) -> Biot-Savart magnetic field computation. 'You are all beneath"
