"""The man who moves a mountain begins by carrying away small stones. -- Confucius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def smith_waterman(
    seq1: str,
    seq2: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -1,
) -> DescriptiveResult:
    """
    Perform local pairwise alignment via Smith-Waterman.

    Dynamic programming with linear gap penalty and zero floor.

    :param seq1: First sequence string.
    :param seq2: Second sequence string.
    :param match: Score for a match. Default 2.
    :param mismatch: Score for a mismatch. Default -1.
    :param gap: Gap penalty (should be <= 0). Default -1.
    :return: DescriptiveResult with best local alignment and score.
    :raises ValueError: If either sequence is empty.

    References
    ----------
    Smith, T. F., & Waterman, M. S. (1981). Identification of common
    molecular subsequences. *J. Mol. Biol.*, 147(1), 195-197.
    """
    if not seq1 or not seq2:
        raise ValueError("Both sequences must be non-empty.")

    m, n = len(seq1), len(seq2)
    H = np.zeros((m + 1, n + 1), dtype=int)

    max_score = 0
    max_pos = (0, 0)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            H[i, j] = max(
                0,
                H[i - 1, j - 1] + s,
                H[i - 1, j] + gap,
                H[i, j - 1] + gap,
            )
            if H[i, j] > max_score:
                max_score = H[i, j]
                max_pos = (i, j)

    align1, align2 = [], []
    i, j = max_pos
    while i > 0 and j > 0 and H[i, j] > 0:
        s = match if seq1[i - 1] == seq2[j - 1] else mismatch
        if H[i, j] == H[i - 1, j - 1] + s:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif H[i, j] == H[i - 1, j] + gap:
            align1.append(seq1[i - 1])
            align2.append("-")
            i -= 1
        else:
            align1.append("-")
            align2.append(seq2[j - 1])
            j -= 1

    a1 = "".join(reversed(align1))
    a2 = "".join(reversed(align2))

    return DescriptiveResult(
        name="Smith-Waterman Alignment",
        value=int(max_score),
        extra={
            "alignment1": a1,
            "alignment2": a2,
            "score": int(max_score),
            "start_pos": (i, j),
            "end_pos": max_pos,
            "length": len(a1),
        },
    )


short = smith_waterman


def cheatsheet() -> str:
    return "smith_waterman({}) -> Smith-Waterman local sequence alignment. 'Great, kid. Don't "
