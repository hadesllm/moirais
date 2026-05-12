# morie.fn -- function file (hadesllm/morie)
"""Simple correspondence analysis. 'Luminous beings are we, not this crude matter.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def correspondence_analysis(table: np.ndarray) -> DescriptiveResult:
    """
    Simple Correspondence Analysis (CA) of a contingency table.

    Decomposes the standardised residual matrix via SVD to extract
    row and column coordinates in a low-dimensional space.

    :param table: (r, c) contingency table of non-negative counts.
    :type table: numpy.ndarray
    :return: DescriptiveResult with row/column coordinates, inertia.
    :rtype: DescriptiveResult
    :raises ValueError: If table is not 2-D or contains negative values.

    References
    ----------
    Greenacre M.J. (1984). *Theory and Applications of Correspondence
    Analysis*. Academic Press.
    """
    T = np.asarray(table, dtype=float)
    if T.ndim != 2:
        raise ValueError(f"Expected 2-D table, got {T.ndim}-D.")
    if np.any(T < 0):
        raise ValueError("Table must contain non-negative values.")
    grand = T.sum()
    if grand == 0:
        raise ValueError("Table sum is zero.")
    P = T / grand
    r_mass = P.sum(axis=1)
    c_mass = P.sum(axis=0)
    Dr_inv = np.diag(1.0 / np.sqrt(np.maximum(r_mass, 1e-12)))
    Dc_inv = np.diag(1.0 / np.sqrt(np.maximum(c_mass, 1e-12)))
    S = Dr_inv @ (P - np.outer(r_mass, c_mass)) @ Dc_inv
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    inertia = sigma**2
    row_coords = Dr_inv @ U * sigma
    col_coords = Dc_inv @ Vt.T * sigma
    return DescriptiveResult(
        name="correspondence_analysis",
        value=float(inertia.sum()),
        extra={
            "row_coords": row_coords,
            "col_coords": col_coords,
            "singular_values": sigma,
            "inertia": inertia,
            "explained_ratio": inertia / inertia.sum() if inertia.sum() > 0 else inertia,
        },
    )


crssp = correspondence_analysis


def cheatsheet() -> str:
    return "correspondence_analysis({}) -> Simple correspondence analysis. 'Luminous beings are we, not"
