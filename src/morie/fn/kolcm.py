# morie.fn -- function file (hadesllm/morie)
"""Kolmogorov complexity approximation via compression."""

import zlib

from ._containers import ESRes


def kolmogorov_complexity(data: bytes | str, **kwargs) -> ESRes:
    """
    Approximate Kolmogorov complexity via zlib compression ratio.

    K(x) is uncomputable, but compression provides an upper bound.
    Normalized complexity C(x) = len(compress(x)) / len(x).

    :param data: bytes or string to measure.
    :return: ESRes with normalized compression complexity.

    References
    ----------
    Li M, Vitanyi P (2008). An Introduction to Kolmogorov Complexity
    and Its Applications, 3rd ed. Springer.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    if len(data) < 1:
        raise ValueError("Data must be non-empty.")
    compressed = zlib.compress(data, level=9)
    raw_len = len(data)
    comp_len = len(compressed)
    ratio = comp_len / raw_len
    return ESRes(
        measure="kolmogorov_complexity",
        estimate=ratio,
        n=raw_len,
        extra={
            "raw_bytes": raw_len,
            "compressed_bytes": comp_len,
            "compression_ratio": ratio,
        },
    )


kolcm = kolmogorov_complexity


def cheatsheet() -> str:
    return "kolmogorov_complexity(data) -> Compression-based complexity."
