"""Geometric anisotropy correction."""

from __future__ import annotations

from ._containers import DescriptiveResult


def anisotropy_correction(coords, ratio, angle):
    """Transform coordinates to correct for geometric anisotropy.

    Rotates by -angle and scales the minor axis by ratio.

    .. epigraph:: It is not the strongest that survives, but the most adaptable. -- Charles Darwin

    Parameters
    ----------
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    ratio : float
        Anisotropy ratio (major/minor range).
    angle : float
        Principal direction in degrees (from x-axis).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    coords = np.asarray(coords, dtype=np.float64)
    theta = np.radians(-angle)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotated = np.column_stack(
        [
            coords[:, 0] * cos_t - coords[:, 1] * sin_t,
            coords[:, 0] * sin_t + coords[:, 1] * cos_t,
        ]
    )

    corrected = rotated.copy()
    corrected[:, 1] *= ratio

    return DescriptiveResult(
        name="anisotropy_correction",
        value=float(ratio),
        extra={
            "corrected_coords": corrected,
            "rotation_angle": angle,
            "ratio": ratio,
            "n_points": coords.shape[0],
        },
    )


sgans = anisotropy_correction


def cheatsheet() -> str:
    return "anisotropy_correction({}) -> Geometric anisotropy correction."
