"""Return wavelet filter bank coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Rebellions are built on hope."


_FILTER_BANK = {
    "haar": np.array([1.0, 1.0]) / np.sqrt(2),
    "db1": np.array([1.0, 1.0]) / np.sqrt(2),
    "db2": np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2),
    "db3": np.array([0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175]) / np.sqrt(2),
    "db4": np.array([0.32580343, 1.01094572, 0.8922014, -0.03957503, -0.26450717, 0.0436163, 0.0465036, -0.01498699])
    / np.sqrt(2),
}


def wavelet_basis(name: str = "db4") -> DescriptiveResult:
    """Return wavelet filter bank coefficients.

    Parameters
    ----------
    name : str
        Wavelet name ('haar', 'db1'-'db4').

    Returns
    -------
    DescriptiveResult
    """
    if name not in _FILTER_BANK:
        raise ValueError(f"Supported wavelets: {sorted(_FILTER_BANK)}, got '{name}'")
    lo = _FILTER_BANK[name].copy()
    hi = np.array([(-1) ** k * lo[len(lo) - 1 - k] for k in range(len(lo))])
    lo_r = lo[::-1]
    hi_r = hi[::-1]
    return DescriptiveResult(
        name="wavelet_basis",
        value=float(len(lo)),
        extra={"lo_d": lo, "hi_d": hi, "lo_r": lo_r, "hi_r": hi_r, "wavelet": name, "length": len(lo)},
    )


wvbas = wavelet_basis


def cheatsheet() -> str:
    return "wavelet_basis({}) -> Return wavelet filter bank coefficients."
