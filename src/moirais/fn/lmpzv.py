# moirais.fn — function file (hadesllm/moirais)
"""Lempel-Ziv complexity."""

__all__ = ["lmpzv"]

import numpy as np
from ._richresult import RichResult


def lmpzv(sequence: np.ndarray) -> dict:
    """
    Compute the Lempel-Ziv complexity of a discrete sequence.

    Counts the number of distinct substrings encountered when scanning
    left to right (LZ76 algorithm). For a random binary sequence of
    length n, the expected complexity approaches n / log2(n).

    Parameters
    ----------
    sequence : np.ndarray
        Integer sequence, shape (n,).

    Returns
    -------
    dict
        'complexity' (int, number of distinct words),
        'normalized' (float, c(n) * log2(n) / n, should approach 1 for
        random sequences),
        'length' (int, sequence length).

    References
    ----------
    Lempel, A. & Ziv, J. (1976). On the complexity of finite sequences.
    IEEE Trans. Inform. Theory, 22(1), 75-81.
    """
    seq = np.asarray(sequence).ravel()
    n = len(seq)
    if n == 0:
        return RichResult(payload={"complexity": 0, "normalized": 0.0, "length": 0})

    s = [int(x) for x in seq]
    complexity = 1
    i = 0
    j = 1
    k = 1
    k_max = 1

    while j + k - 1 < n:
        if s[i + k - 1] == s[j + k - 1]:
            k += 1
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == j:
                complexity += 1
                j += k_max
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    if k != 1 or i != 0:
        complexity += 1

    if n > 1:
        normalized = complexity * np.log2(n) / n
    else:
        normalized = float(complexity)

    return RichResult(payload={"complexity": complexity, "normalized": normalized, "length": n})
