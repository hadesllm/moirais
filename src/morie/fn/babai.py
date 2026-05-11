# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Babai's nearest plane algorithm for CVP."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def babai_cvp(basis: np.ndarray, target: np.ndarray) -> DescriptiveResult:
    """Approximate Closest Vector Problem via Babai's nearest plane.

    :param basis: n x m matrix (rows are basis vectors).
    :param target: target vector (length m).
    :return: DescriptiveResult with closest vector in ``extra``.
    """
    from morie.crypto._lattice_core import babai_nearest_plane as _babai

    B = np.asarray(basis, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    closest = _babai(B, t)
    dist = float(np.linalg.norm(t - closest))
    return DescriptiveResult(
        name="babai_cvp",
        value=dist,
        extra={
            "closest_vector": closest,
            "distance": dist,
            "target": t.tolist(),
            "dimension": B.shape[0],
        },
    )


babai = babai_cvp


def cheatsheet() -> str:
    return "babai_cvp({}) -> Babai's nearest plane algorithm for CVP."
