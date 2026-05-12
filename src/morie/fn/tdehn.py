"""Dehn twist on the torus."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dehn_twist(curve_type: str = "a", n: int = 1) -> DescriptiveResult:
    """Knowledge itself is power. -- Francis Bacon"""
    curve_type = curve_type.lower()
    if curve_type not in ("a", "b"):
        raise ValueError(f"curve_type must be 'a' or 'b', got '{curve_type}'.")
    if curve_type == "a":
        M = np.array([[1, 0], [n, 1]], dtype=int)
    else:
        M = np.array([[1, n], [0, 1]], dtype=int)
    tr = int(M[0, 0] + M[1, 1])
    classification = "parabolic" if n != 0 else "identity"
    return DescriptiveResult(
        name="dehn_twist",
        value=float(n),
        extra={
            "matrix": M.tolist(),
            "curve_type": curve_type,
            "n_twists": n,
            "trace": tr,
            "classification": classification,
        },
    )


def cheatsheet() -> str:
    return "dehn_twist(curve_type, n) -> Dehn twist matrix on torus"
