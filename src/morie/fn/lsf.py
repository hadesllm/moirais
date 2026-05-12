# morie.fn -- function file (hadesllm/morie)
"""Line Spectral Frequencies from LPC coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You underestimate my power!"


def line_spectral_freq_fn(lpc_coeffs: np.ndarray) -> DescriptiveResult:
    """Compute Line Spectral Frequencies (LSF) from LPC coefficients.

    Decomposes the LPC polynomial :math:`A(z)` into symmetric and
    antisymmetric polynomials :math:`P(z)` and :math:`Q(z)`, whose
    roots on the unit circle yield LSFs.

    :param lpc_coeffs: LPC coefficients [a1, ..., ap].
    :return: DescriptiveResult with LSF values in radians.
    """
    lpc_coeffs = np.asarray(lpc_coeffs, dtype=float).ravel()
    p = len(lpc_coeffs)
    a = np.concatenate(([1.0], -lpc_coeffs))
    p_poly = a + a[::-1]
    q_poly = a - a[::-1]
    p_roots = np.roots(p_poly)
    q_roots = np.roots(q_poly)
    p_angles = np.sort(np.abs(np.angle(p_roots[np.angle(p_roots) > 0])))
    q_angles = np.sort(np.abs(np.angle(q_roots[np.angle(q_roots) > 0])))
    lsf = np.sort(np.concatenate([p_angles, q_angles]))[:p]
    return DescriptiveResult(
        name="line_spectral_freq",
        value=None,
        extra={"lsf": lsf, "order": p},
    )


lsf = line_spectral_freq_fn


def cheatsheet() -> str:
    return "line_spectral_freq_fn({}) -> Line Spectral Frequencies from LPC coefficients."
