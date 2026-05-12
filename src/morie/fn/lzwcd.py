# morie.fn -- function file (hadesllm/morie)
"""LZW compression (Lempel-Ziv-Welch)."""

__all__ = ["lzwcd"]

import numpy as np


def lzwcd(data: np.ndarray, *, alphabet_size: int = 256) -> dict:
    """
    LZW compression for integer sequences.

    Parameters
    ----------
    data : np.ndarray
        Integer array with values in [0, alphabet_size).
    alphabet_size : int
        Size of the base alphabet. Default 256 (byte-level).

    Returns
    -------
    dict
        'compressed' (list of int codes),
        'compression_ratio' (original_bits / compressed_bits),
        'dictionary_size' (final dictionary size),
        'decompressed' (np.ndarray, round-trip verification).

    Raises
    ------
    ValueError
        If data contains values outside [0, alphabet_size).

    References
    ----------
    Welch, T. (1984). A technique for high-performance data compression.
    Computer, 17(6), 8-19.
    """
    data = np.asarray(data, dtype=np.int64).ravel()
    if len(data) == 0:
        return {"compressed": [], "compression_ratio": 1.0,
                "dictionary_size": alphabet_size, "decompressed": np.array([], dtype=np.int64)}

    if np.any(data < 0) or np.any(data >= alphabet_size):
        raise ValueError(f"data values must be in [0, {alphabet_size}).")

    dictionary = {(i,): i for i in range(alphabet_size)}
    next_code = alphabet_size
    compressed = []
    w = (int(data[0]),)

    for i in range(1, len(data)):
        c = int(data[i])
        wc = w + (c,)
        if wc in dictionary:
            w = wc
        else:
            compressed.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = (c,)
    compressed.append(dictionary[w])

    orig_bits = len(data) * int(np.ceil(np.log2(max(alphabet_size, 2))))
    code_bits = max(int(np.ceil(np.log2(max(next_code, 2)))), 1)
    comp_bits = len(compressed) * code_bits
    ratio = orig_bits / comp_bits if comp_bits > 0 else float("inf")

    inv_dict = {i: (i,) for i in range(alphabet_size)}
    decompressed = []
    w_dec = (compressed[0],)
    decompressed.extend(w_dec)
    inv_next = alphabet_size

    for code in compressed[1:]:
        if code in inv_dict:
            entry = inv_dict[code]
        elif code == inv_next:
            entry = w_dec + (w_dec[0],)
        else:
            raise ValueError(f"Bad compressed code: {code}")
        decompressed.extend(entry)
        inv_dict[inv_next] = w_dec + (entry[0],)
        inv_next += 1
        w_dec = entry

    return {
        "compressed": compressed,
        "compression_ratio": ratio,
        "dictionary_size": next_code,
        "decompressed": np.array(decompressed, dtype=np.int64),
    }
