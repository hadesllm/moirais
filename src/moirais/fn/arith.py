# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Arithmetic coding (encode and decode)."""

__all__ = ["arith"]

import numpy as np


def arith(message: list, pmf: np.ndarray, symbols: list = None, cdf=None, *, precision: int = 52) -> dict:
    """
    Arithmetic coding: encode a message into a sub-interval of [0, 1).

    Parameters
    ----------
    message : list
        Sequence of symbols to encode.
    pmf : np.ndarray
        Symbol probabilities, shape (n,). Must sum to 1.
    symbols : list, optional
        Symbol labels matching pmf. Defaults to 0..n-1.
    precision : int
        Bits of floating-point precision for the interval.

    Returns
    -------
    dict
        'code_value' (float in [0, 1)),
        'interval' (low, high),
        'bits_required' (int, ceil(-log2(high - low))),
        'decoded' (list, decoded message to verify round-trip).

    Raises
    ------
    ValueError
        If pmf invalid or message contains unknown symbols.

    References
    ----------
    Rissanen, J. & Langdon, G. (1979). Arithmetic coding.
    IBM J. Res. Dev., 23(2), 149-162.
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if pmf.ndim != 1 or not np.isclose(pmf.sum(), 1.0):
        raise ValueError("pmf must be a 1-D array summing to 1.")

    n = len(pmf)
    if symbols is None:
        symbols = list(range(n))
    if len(symbols) != n:
        raise ValueError("symbols length must match pmf length.")

    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    for s in message:
        if s not in sym_to_idx:
            raise ValueError(f"Symbol {s!r} not in symbols list.")

    cdf = np.zeros(n + 1)
    for i in range(n):
        cdf[i + 1] = cdf[i] + pmf[i]

    low, high = 0.0, 1.0
    for s in message:
        idx = sym_to_idx[s]
        width = high - low
        high = low + width * cdf[idx + 1]
        low = low + width * cdf[idx]

    code_value = (low + high) / 2.0
    interval_width = high - low
    bits_req = int(np.ceil(-np.log2(interval_width))) if interval_width > 0 else 0

    decoded = []
    val = code_value
    lo, hi = 0.0, 1.0
    for _ in range(len(message)):
        width = hi - lo
        scaled = (val - lo) / width
        idx = 0
        for j in range(n):
            if cdf[j] <= scaled < cdf[j + 1]:
                idx = j
                break
        decoded.append(symbols[idx])
        hi = lo + width * cdf[idx + 1]
        lo = lo + width * cdf[idx]

    return {
        "code_value": code_value,
        "interval": (low, high),
        "bits_required": bits_req,
        "decoded": decoded,
    }
