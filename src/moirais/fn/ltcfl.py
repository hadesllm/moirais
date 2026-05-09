# moirais.fn — function file (hadesllm/moirais)
"""Lattice filter coefficients from AR coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


def lattice_coefficients_fn(ar_coeffs: np.ndarray) -> DescriptiveResult:
    """Convert AR coefficients to lattice filter (reflection) coefficients.

    The lattice structure parameterises the all-pole filter as a cascade
    of two-multiplier sections with reflection coefficients :math:`k_m`.

    :param ar_coeffs: 1-D array of AR coefficients [a1, ..., ap].
    :return: DescriptiveResult with lattice coefficients.
    """
    from moirais._armodel import reflection_coefficients

    ar_coeffs = np.asarray(ar_coeffs, dtype=float).ravel()
    k = reflection_coefficients(ar_coeffs)
    return DescriptiveResult(
        name="lattice_coefficients",
        value=None,
        extra={"lattice_coeffs": k, "order": len(ar_coeffs)},
    )


ltcfl = lattice_coefficients_fn


def cheatsheet() -> str:
    return "lattice_coefficients_fn({}) -> Lattice filter coefficients from AR coefficients."
