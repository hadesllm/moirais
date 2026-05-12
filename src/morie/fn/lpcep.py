# morie.fn -- function file (hadesllm/morie)
"""Convert LPC coefficients to cepstral coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "These aren't the droids you're looking for."


def lpc_to_cepstral_fn(lpc_coeffs: np.ndarray, n_ceps: int = 13) -> DescriptiveResult:
    r"""Convert LPC coefficients to cepstral coefficients.

    The recursion from AR to cepstral domain:

    .. math::

        c(n) = a(n) + \\sum_{k=1}^{n-1} \\frac{k}{n} c(k) a(n-k), \\quad 1 \\le n \\le p

    :param lpc_coeffs: LPC coefficients [a1, ..., ap].
    :param n_ceps: Number of cepstral coefficients to produce (default 13).
    :return: DescriptiveResult with cepstral coefficients.
    """
    from morie._armodel import cepstral_coefficients

    lpc_coeffs = np.asarray(lpc_coeffs, dtype=float).ravel()
    ceps = cepstral_coefficients(lpc_coeffs, n_coeffs=n_ceps)
    return DescriptiveResult(
        name="lpc_to_cepstral",
        value=None,
        extra={"cepstral_coeffs": ceps, "n_ceps": n_ceps, "order": len(lpc_coeffs)},
    )


lpcep = lpc_to_cepstral_fn


def cheatsheet() -> str:
    return "lpc_to_cepstral_fn({}) -> Convert LPC coefficients to cepstral coefficients."
