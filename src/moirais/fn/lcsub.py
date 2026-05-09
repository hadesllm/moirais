# moirais.fn — function file (hadesllm/moirais)
"""Longest common subsequence. 'Patience you must have.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def longest_common_subseq(
    seq1: str,
    seq2: str,
) -> DescriptiveResult:
    """
    Find the longest common subsequence via dynamic programming.

    .. math::

        LCS[i,j] = \\begin{cases}
        LCS[i-1,j-1]+1 & \\text{if } s_1[i]=s_2[j] \\\\
        \\max(LCS[i-1,j], LCS[i,j-1]) & \\text{otherwise}
        \\end{cases}

    :param seq1: First sequence (string or list).
    :param seq2: Second sequence (string or list).
    :return: DescriptiveResult with LCS string and length.
    :raises ValueError: If either sequence is empty.

    References
    ----------
    Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
    (2009). *Introduction to Algorithms*. 3rd ed. MIT Press. Ch. 15.
    """
    if not seq1 or not seq2:
        raise ValueError("Both sequences must be non-empty.")

    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    lcs_chars = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs_chars.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1, j] > dp[i, j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_str = "".join(reversed(lcs_chars))

    return DescriptiveResult(
        name="Longest Common Subsequence",
        value=int(dp[m, n]),
        extra={
            "lcs": lcs_str,
            "length": int(dp[m, n]),
            "similarity": float(dp[m, n]) / max(m, n),
            "len_seq1": m,
            "len_seq2": n,
        },
    )


short = longest_common_subseq


def cheatsheet() -> str:
    return "longest_common_subseq({}) -> Longest common subsequence. 'Patience you must have.' -- Yod"
