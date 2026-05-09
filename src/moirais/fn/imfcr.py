# moirais.fn — function file (hadesllm/moirais)
"""IMF stopping criteria check."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Size matters not. Judge me by my size, do you?"


def imf_criteria(imf, x, **kwargs) -> DescriptiveResult:
    """Check IMF stopping criteria: Cauchy, S-number, energy ratio.

    Evaluates whether *imf* is a valid Intrinsic Mode Function relative
    to the original signal *x*.

    Parameters
    ----------
    imf : array-like
        Candidate IMF.
    x : array-like
        Original signal.

    Returns
    -------
    DescriptiveResult
        ``value`` is True if all criteria pass; ``extra`` has
        ``cauchy``, ``s_number``, ``energy_ratio``, ``zero_crossings``,
        ``n_extrema``.

    References
    ----------
    Huang, N. E., et al. (1998). The empirical mode decomposition and
    the Hilbert spectrum. *Proc. R. Soc. Lond. A*, 454, 903-995.
    """
    imf = np.asarray(imf, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    N = len(imf)

    zero_crossings = int(np.sum(np.diff(np.sign(imf)) != 0))
    n_extrema = 0
    for i in range(1, N - 1):
        if imf[i] > imf[i - 1] and imf[i] >= imf[i + 1]:
            n_extrema += 1
        if imf[i] < imf[i - 1] and imf[i] <= imf[i + 1]:
            n_extrema += 1

    s_number = abs(zero_crossings - n_extrema)

    residual = x - imf
    energy_ratio = float(np.sum(residual**2) / (np.sum(x**2) + 1e-15))

    cauchy_sd = float(np.sum((imf - np.mean(imf)) ** 2) / (N + 1e-15))
    cauchy_ok = cauchy_sd > 0

    all_pass = (s_number <= 1) and (energy_ratio < 1.0) and cauchy_ok

    return DescriptiveResult(
        name="imf_criteria",
        value=all_pass,
        extra={
            "cauchy": cauchy_ok,
            "s_number": s_number,
            "energy_ratio": energy_ratio,
            "zero_crossings": zero_crossings,
            "n_extrema": n_extrema,
        },
    )


imfcr = imf_criteria


def cheatsheet() -> str:
    return "imf_criteria({}) -> IMF stopping criteria check."
