# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""AR coefficients to line spectral frequencies."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Mastering others is strength; mastering yourself is true power. — Lao Tzu"


def ar_to_lsf(ar_coeffs, **kwargs) -> DescriptiveResult:
    """Convert AR coefficients to line spectral frequencies (LSF).

    Decomposes A(z) into symmetric P(z) and antisymmetric Q(z) polynomials,
    then finds their roots on the unit circle.

    Parameters
    ----------
    ar_coeffs : array-like
        AR coefficients (with leading 1).

    Returns
    -------
    DescriptiveResult
    """
    a = np.asarray(ar_coeffs, dtype=float)
    if abs(a[0]) > 0:
        a = a / a[0]
    p = len(a) - 1
    a1 = np.concatenate((a, [0]))
    a2 = a1[::-1]
    P = a1 + a2
    Q = a1 - a2
    rP = np.roots(P)
    rQ = np.roots(Q)
    anglesP = np.angle(rP[np.abs(np.abs(rP) - 1.0) < 0.02])
    anglesQ = np.angle(rQ[np.abs(np.abs(rQ) - 1.0) < 0.02])
    anglesP = np.sort(anglesP[anglesP > 0])
    anglesQ = np.sort(anglesQ[anglesQ > 0])
    lsf = np.sort(np.concatenate((anglesP, anglesQ)))[:p]
    return DescriptiveResult(
        name="ar_to_lsf",
        value=float(lsf[0]) if len(lsf) > 0 else 0.0,
        extra={"lsf": lsf, "ar": ar_coeffs, "order": p},
    )


ar2ls = ar_to_lsf


def cheatsheet() -> str:
    return "ar_to_lsf({}) -> AR coefficients to line spectral frequencies."
