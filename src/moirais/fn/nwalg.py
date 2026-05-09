# moirais.fn — function file (hadesllm/moirais)
"""Needleman-Wunsch global sequence alignment. 'He who fights with monsters should be careful lest he thereby become a monster. — Friedrich Nietzsche' -- Chirrut Imwe"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def needleman_wunsch(
    seq1: str,
    seq2: str,
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
) -> DescriptiveResult:
    """
    Perform global pairwise alignment via Needleman-Wunsch.

    Dynamic programming with linear gap penalty.

    :param seq1: First sequence string.
    :param seq2: Second sequence string.
    :param match: Score for a match. Default 1.
    :param mismatch: Score for a mismatch. Default -1.
    :param gap: Gap penalty (should be <= 0). Default -1.
    :return: DescriptiveResult with aligned sequences and score.
    :raises ValueError: If either sequence is empty.

    References
    ----------
    Needleman, S. B., & Wunsch, C. D. (1970). A general method
    applicable to the search for similarities in the amino acid
    sequence of two proteins. *J. Mol. Biol.*, 48(3), 443-453.
    """
    if not seq1 or not seq2:
        raise ValueError("Both sequences must be non-empty.")

    m, n = len(seq1), len(seq2)
    score_mat = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        score_mat[i, 0] = i * gap
    for j in range(1, n + 1):
        score_mat[0, j] = j * gap

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            score_mat[i, j] = max(
                score_mat[i - 1, j - 1] + s,
                score_mat[i - 1, j] + gap,
                score_mat[i, j - 1] + gap,
            )

    align1, align2 = [], []
    i, j = m, n
    while i > 0 and j > 0:
        s = match if seq1[i - 1] == seq2[j - 1] else mismatch
        if score_mat[i, j] == score_mat[i - 1, j - 1] + s:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif score_mat[i, j] == score_mat[i - 1, j] + gap:
            align1.append(seq1[i - 1])
            align2.append("-")
            i -= 1
        else:
            align1.append("-")
            align2.append(seq2[j - 1])
            j -= 1
    while i > 0:
        align1.append(seq1[i - 1])
        align2.append("-")
        i -= 1
    while j > 0:
        align1.append("-")
        align2.append(seq2[j - 1])
        j -= 1

    a1 = "".join(reversed(align1))
    a2 = "".join(reversed(align2))
    identity = sum(1 for c1, c2 in zip(a1, a2) if c1 == c2) / len(a1) * 100

    return DescriptiveResult(
        name="Needleman-Wunsch Alignment",
        value=int(score_mat[m, n]),
        extra={
            "alignment1": a1,
            "alignment2": a2,
            "score": int(score_mat[m, n]),
            "identity_pct": float(identity),
            "length": len(a1),
        },
    )


short = needleman_wunsch


def cheatsheet() -> str:
    return "Mastering others is strength; mastering yourself is true power. — Lao Tzu"
