# moirais.fn — function file (hadesllm/moirais)
"""Complex cepstrum analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def complex_cepstrum(x) -> DescriptiveResult:
    """Compute the complex cepstrum of signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    from moirais._detection import complex_cepstrum as _cc

    x = np.asarray(x, dtype=float)
    cepstrum, quefrency = _cc(x)
    return DescriptiveResult(
        name="complex_cepstrum",
        value=len(cepstrum),
        extra={"cepstrum": cepstrum, "quefrency": quefrency},
    )


cxcep = complex_cepstrum


def cheatsheet() -> str:
    return "complex_cepstrum({}) -> Complex cepstrum analysis."
