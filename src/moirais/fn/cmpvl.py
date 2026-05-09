# moirais.fn — function file (hadesllm/moirais)
"""Violation type distribution."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import DescriptiveResult


def compliance_violation(
    violation_types: np.ndarray,
) -> DescriptiveResult:
    """Distribution of violation types.

    Parameters
    ----------
    violation_types : ndarray
        Violation type labels.

    Returns
    -------
    DescriptiveResult
    """
    vt = np.asarray(violation_types)
    unique, counts = np.unique(vt, return_counts=True)
    n = len(vt)
    dist = {str(u): int(c) for u, c in zip(unique, counts)}
    props = {str(u): float(c / n) for u, c in zip(unique, counts)}
    return DescriptiveResult(
        name="compliance_violation", value=None, extra={"distribution": dist, "proportions": props, "n": n}
    )


cmpvl = compliance_violation


def cheatsheet() -> str:
    return "compliance_violation({}) -> Violation type distribution."
