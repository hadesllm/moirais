# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Simplified BLOSUM62 alignment score. 'Judge me by my size, do you?'"""

from __future__ import annotations

from ._containers import DescriptiveResult

_BLOSUM62_COMMON = {
    ("A", "A"): 4,
    ("R", "R"): 5,
    ("N", "N"): 6,
    ("D", "D"): 6,
    ("C", "C"): 9,
    ("Q", "Q"): 5,
    ("E", "E"): 5,
    ("G", "G"): 6,
    ("H", "H"): 8,
    ("I", "I"): 4,
    ("L", "L"): 4,
    ("K", "K"): 5,
    ("M", "M"): 5,
    ("F", "F"): 6,
    ("P", "P"): 7,
    ("S", "S"): 4,
    ("T", "T"): 5,
    ("W", "W"): 11,
    ("Y", "Y"): 7,
    ("V", "V"): 4,
}

_BLOSUM62_MISMATCH_DEFAULT = -1


def blosum_score(
    seq1: str,
    seq2: str,
    gap_penalty: int = -4,
) -> DescriptiveResult:
    """
    Compute a simplified BLOSUM62-like alignment score.

    Uses diagonal BLOSUM62 self-scores for matches, a default mismatch
    penalty, and a linear gap penalty. Sequences must be equal length
    (pre-aligned).

    :param seq1: First aligned amino acid sequence (uppercase).
    :param seq2: Second aligned amino acid sequence (uppercase, same length).
    :param gap_penalty: Penalty for gap characters ('-'). Default -4.
    :return: DescriptiveResult with total score and per-position breakdown.
    :raises ValueError: If sequences differ in length.

    References
    ----------
    Henikoff, S., & Henikoff, J. G. (1992). Amino acid substitution
    matrices from protein blocks. *PNAS*, 89(22), 10915-10919.
    """
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must be equal length: {len(seq1)} vs {len(seq2)}.")
    if not seq1:
        raise ValueError("Sequences must be non-empty.")

    s1 = seq1.upper()
    s2 = seq2.upper()

    position_scores = []
    for c1, c2 in zip(s1, s2):
        if c1 == "-" or c2 == "-":
            position_scores.append(gap_penalty)
        elif c1 == c2:
            position_scores.append(_BLOSUM62_COMMON.get((c1, c1), 1))
        else:
            pair = tuple(sorted((c1, c2)))
            position_scores.append(_BLOSUM62_COMMON.get(pair, _BLOSUM62_MISMATCH_DEFAULT))

    total = sum(position_scores)
    n_match = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2 and c1 != "-")
    n_gap = sum(1 for c1, c2 in zip(s1, s2) if c1 == "-" or c2 == "-")

    return DescriptiveResult(
        name="BLOSUM62 Score",
        value=total,
        extra={
            "position_scores": position_scores,
            "total_score": total,
            "n_matches": n_match,
            "n_gaps": n_gap,
            "identity_pct": float(n_match / len(s1) * 100) if s1 else 0.0,
            "length": len(s1),
        },
    )


short = blosum_score


def cheatsheet() -> str:
    return "blosum_score({}) -> Simplified BLOSUM62 alignment score. 'Judge me by my size, d"
