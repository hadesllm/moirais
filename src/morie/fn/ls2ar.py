# morie.fn — function file (hadesllm/morie)
"""Line spectral frequencies to AR coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Hope is like the sun. If you only believe in it when you see it, you'll never make it through the night."


def lsf_to_ar(lsf, **kwargs) -> DescriptiveResult:
    """Convert line spectral frequencies back to AR coefficients.

    Parameters
    ----------
    lsf : array-like
        Line spectral frequencies in (0, pi).

    Returns
    -------
    DescriptiveResult
    """
    lsf = np.asarray(lsf, dtype=float)
    p = len(lsf)
    P_roots = []
    Q_roots = []
    for i, w in enumerate(lsf):
        if i % 2 == 0:
            P_roots.extend([np.exp(1j * w), np.exp(-1j * w)])
        else:
            Q_roots.extend([np.exp(1j * w), np.exp(-1j * w)])
    if len(P_roots) == 0:
        P = np.array([1.0])
    else:
        P = np.real(np.poly(P_roots))
    if len(Q_roots) == 0:
        Q = np.array([1.0])
    else:
        Q_roots.extend([1.0, -1.0])
        Q = np.real(np.poly(Q_roots))
    maxlen = max(len(P), len(Q))
    P = np.pad(P, (0, maxlen - len(P)))
    Q = np.pad(Q, (0, maxlen - len(Q)))
    a = 0.5 * (P + Q)
    a = a[: p + 1]
    if abs(a[0]) > 0:
        a = a / a[0]
    return DescriptiveResult(
        name="lsf_to_ar",
        value=float(a[1]) if len(a) > 1 else 0.0,
        extra={"ar": a, "lsf": lsf},
    )


ls2ar = lsf_to_ar


def cheatsheet() -> str:
    return "lsf_to_ar({}) -> Line spectral frequencies to AR coefficients."
