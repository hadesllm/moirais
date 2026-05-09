# moirais.fn — function file (hadesllm/moirais)
"""Lempel-Ziv complexity."""

import numpy as np

from ._containers import ESRes


def lempel_ziv_complexity(x, threshold: float | None = None, **kwargs) -> ESRes:
    """
    Compute Lempel-Ziv complexity of a binary sequence.

    Counts the number of distinct substrings encountered when parsing
    left to right (LZ76 algorithm). For continuous data, binarises
    at the median (or given threshold).

    :param x: array-like. Continuous data is binarised.
    :param threshold: Binarisation threshold. Default: median.
    :return: ESRes with LZ complexity and normalised complexity.

    References
    ----------
    Lempel A, Ziv J (1976). On the complexity of finite sequences.
    IEEE Transactions on Information Theory, 22(1), 75-81.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 1:
        raise ValueError("Need at least 1 observation.")
    if threshold is None:
        threshold = float(np.median(x))
    s = "".join("1" if v > threshold else "0" for v in x)

    complexity = 1
    i = 0
    k = 1
    kmax = 1
    while i + k <= n:
        sub = s[i + 1 : i + k + 1] if i + k + 1 <= n else ""
        if s[i + k : i + k + 1] == "" or sub == "":
            break
        if s[i + k] in s[i:i + k]:
            k += 1
            kmax = max(kmax, k)
        else:
            complexity += 1
            i += kmax if kmax > 0 else 1
            k = 1
            kmax = 1

    norm = complexity * np.log2(n) / n if n > 1 else 0.0
    return ESRes(
        measure="lempel_ziv_complexity",
        estimate=float(complexity),
        n=n,
        extra={"normalised": float(norm), "threshold": threshold},
    )


lzcmp = lempel_ziv_complexity


def cheatsheet() -> str:
    return "lempel_ziv_complexity(x) -> Lempel-Ziv complexity."
