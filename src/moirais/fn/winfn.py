"""Window functions for spectral analysis."""

from __future__ import annotations

from ._containers import DescriptiveResult


def window_functions(N: int, wtype: str = "hamming") -> DescriptiveResult:
    """Generate a spectral window of specified type.

    'This is the way.' — The Mandalorian
    """
    from moirais._spectral import window_functions as _backend

    w = _backend(N, wtype=wtype)
    return DescriptiveResult(
        name="window_function",
        value=len(w),
        extra={"window": w, "wtype": wtype, "N": N},
    )


winfn = window_functions


def cheatsheet() -> str:
    return "window_functions({}) -> Window functions for spectral analysis."
