# morie.fn — function file (hadesllm/morie)
"""Reflection (PARCOR) coefficients from AR coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your eyes can deceive you. Don't trust them."


def reflection_coefficients_fn(ar_coeffs: np.ndarray) -> DescriptiveResult:
    """Convert AR polynomial coefficients to reflection (PARCOR) coefficients.

    Uses step-down recursion from the AR polynomial :math:`A(z)` to extract
    the lattice reflection coefficients :math:`k_i`.

    :param ar_coeffs: 1-D array of AR coefficients [a1, a2, ..., ap].
    :return: DescriptiveResult with reflection coefficients.
    """
    from morie._armodel import reflection_coefficients

    ar_coeffs = np.asarray(ar_coeffs, dtype=float).ravel()
    k = reflection_coefficients(ar_coeffs)
    stable = bool(np.all(np.abs(k) < 1))
    return DescriptiveResult(
        name="reflection_coefficients",
        value=None,
        extra={"reflection_coeffs": k, "stable": stable, "order": len(ar_coeffs)},
    )


rflcf = reflection_coefficients_fn


def cheatsheet() -> str:
    return "reflection_coefficients_fn({}) -> Reflection (PARCOR) coefficients from AR coefficients."
