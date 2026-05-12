# morie.fn -- function file (hadesllm/morie)
"""Conditional mutual information I(X;Y|Z)."""

import numpy as np

from ._containers import ESRes


def conditional_mutual_information(x, y, z, bins: int = 20, **kwargs) -> ESRes:
    """
    Compute conditional mutual information I(X;Y|Z).

    .. math::

        I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    Uses histogram-based estimation with 3-D binning.

    :param x: array-like, first variable.
    :param y: array-like, second variable.
    :param z: array-like, conditioning variable.
    :param bins: Number of bins per dimension.
    :return: ESRes with conditional MI in bits.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory,
    2nd ed. Wiley, New York.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    n = len(x)
    if n != len(y) or n != len(z):
        raise ValueError("x, y, z must have same length.")
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    def _h(data):
        hist, _ = np.histogramdd(data, bins=bins)
        p = hist / hist.sum()
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    h_xz = _h(np.column_stack([x, z]))
    h_yz = _h(np.column_stack([y, z]))
    h_xyz = _h(np.column_stack([x, y, z]))
    h_z = _h(z.reshape(-1, 1))

    cmi = h_xz + h_yz - h_xyz - h_z
    return ESRes(
        measure="conditional_mutual_information",
        estimate=cmi,
        n=n,
        extra={"H_XZ": h_xz, "H_YZ": h_yz, "H_XYZ": h_xyz, "H_Z": h_z},
    )


cmifn = conditional_mutual_information


def cheatsheet() -> str:
    return "conditional_mutual_information(x, y, z) -> I(X;Y|Z)."
