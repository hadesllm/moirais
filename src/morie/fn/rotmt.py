# morie.fn — function file (hadesllm/morie)
"""2D rotation matrix. 'Dismantle.' -- Sukuna, Jujutsu Kaisen"""

from __future__ import annotations

from ._containers import DescriptiveResult


def rotation_matrix_2d(theta):
    """Construct a 2x2 rotation matrix for angle theta (radians).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    DescriptiveResult
        value = 2x2 rotation matrix.
    """
    import numpy as np

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return DescriptiveResult(name="rotation_matrix_2d", value=R, extra={"theta": float(theta)})


rotmt = rotation_matrix_2d


def cheatsheet() -> str:
    return "rotation_matrix_2d({}) -> 2D rotation matrix. 'Dismantle.' -- Sukuna, Jujutsu Kaisen"
