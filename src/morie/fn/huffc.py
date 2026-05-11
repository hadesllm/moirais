# morie.fn — function file (hadesllm/morie)
"""Huffman coding (optimal prefix code)."""

__all__ = ["huffc"]

import heapq

import numpy as np


def huffc(pmf: np.ndarray, symbols: list = None) -> dict:
    """
    Build a Huffman code for a discrete source.

    Parameters
    ----------
    pmf : np.ndarray
        Symbol probabilities, shape (n,). Must sum to 1.
    symbols : list, optional
        Symbol labels. Defaults to 0..n-1.

    Returns
    -------
    dict
        'codebook' (dict: symbol -> binary string),
        'avg_length' (expected codeword length in bits),
        'entropy' (source entropy in bits),
        'efficiency' (entropy / avg_length).

    Raises
    ------
    ValueError
        If pmf invalid or symbols length mismatch.

    References
    ----------
    Huffman, D. (1952). A method for the construction of minimum-redundancy
    codes. Proc. IRE, 40(9), 1098-1101.
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if pmf.ndim != 1 or not np.isclose(pmf.sum(), 1.0):
        raise ValueError("pmf must be a 1-D array summing to 1.")
    if np.any(pmf < 0):
        raise ValueError("pmf entries must be non-negative.")

    n = len(pmf)
    if symbols is None:
        symbols = list(range(n))
    if len(symbols) != n:
        raise ValueError("symbols length must match pmf length.")

    if n == 1:
        codebook = {symbols[0]: "0"}
    else:
        heap = []
        for i, p in enumerate(pmf):
            heapq.heappush(heap, (p, i, symbols[i]))

        codes = {s: "" for s in symbols}
        counter = n

        while len(heap) > 1:
            p1, _, node1 = heapq.heappop(heap)
            p2, _, node2 = heapq.heappop(heap)

            if isinstance(node1, list):
                for s in node1:
                    codes[s] = "0" + codes[s]
            else:
                codes[node1] = "0" + codes[node1]
                node1 = [node1]

            if isinstance(node2, list):
                for s in node2:
                    codes[s] = "1" + codes[s]
            else:
                codes[node2] = "1" + codes[node2]
                node2 = [node2]

            merged = node1 + node2
            heapq.heappush(heap, (p1 + p2, counter, merged))
            counter += 1

        codebook = codes

    avg_len = sum(pmf[i] * len(codebook[symbols[i]]) for i in range(n))
    eps = 1e-300
    entropy = -np.sum(pmf[pmf > 0] * np.log2(pmf[pmf > 0]))
    efficiency = entropy / avg_len if avg_len > 0 else 0.0

    return {
        "codebook": codebook,
        "avg_length": avg_len,
        "entropy": entropy,
        "efficiency": efficiency,
    }
