# morie.fn -- function file (hadesllm/morie)
"""Computable complexity ratio."""

import zlib

from ._containers import ESRes


def computable_complexity(data: bytes | str, **kwargs) -> ESRes:
    """
    Compute the computable complexity ratio (CCR).

    Ratio of compressed size to theoretical minimum (log2 of alphabet
    size * length). Values near 1 indicate incompressible (random) data.

    :param data: bytes or string.
    :return: ESRes with complexity ratio.

    References
    ----------
    Li M, Vitanyi P (2008). An Introduction to Kolmogorov Complexity
    and Its Applications, 3rd ed. Springer.
    """
    if isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = data
    if len(raw) < 1:
        raise ValueError("Data must be non-empty.")
    compressed = zlib.compress(raw, level=9)
    alphabet_size = len(set(raw))
    if alphabet_size < 2:
        theoretical_min = len(raw)
    else:
        import math
        theoretical_min = len(raw) * math.log2(alphabet_size) / 8.0
    ratio = len(compressed) / max(theoretical_min, 1.0)
    return ESRes(
        measure="computable_complexity",
        estimate=float(ratio),
        n=len(raw),
        extra={
            "compressed_bytes": len(compressed),
            "raw_bytes": len(raw),
            "alphabet_size": alphabet_size,
        },
    )


cmpls = computable_complexity


def cheatsheet() -> str:
    return "computable_complexity(data) -> Complexity ratio."
