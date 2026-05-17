"""Detect palindromic subsequences in a numeric or string sequence."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def palindrome_detect(
    sequence: np.ndarray | list,
    *,
    min_length: int = 3,
    tolerance: float = 0.0,
) -> DescriptiveResult:
    """Detect palindromic subsequences in a numeric or string sequence.

    Finds all maximal palindromic runs.  For numeric sequences, uses
    approximate matching within *tolerance*.  Useful in genomics (palindromic
    restriction sites), signal processing (symmetric patterns), and time
    series analysis.

    Parameters
    ----------
    sequence : array or list
        Input sequence.
    min_length : int
        Minimum palindrome length to report.
    tolerance : float
        For numeric data, maximum element-wise difference for approximate match.

    Returns
    -------
    DescriptiveResult
        ``value`` = number of palindromic subsequences found.
    """
    seq = list(sequence)
    n = len(seq)
    if n < min_length:
        raise ValueError(f"Sequence length {n} < min_length {min_length}")
    is_numeric = all(isinstance(s, (int, float, np.integer, np.floating)) for s in seq)
    palindromes = []
    for center in range(n):
        for parity in (0, 1):
            lo = center
            hi = center + parity
            while lo >= 0 and hi < n:
                if is_numeric:
                    if abs(float(seq[lo]) - float(seq[hi])) > tolerance:
                        break
                else:
                    if seq[lo] != seq[hi]:
                        break
                lo -= 1
                hi += 1
            lo += 1
            hi -= 1
            length = hi - lo + 1
            if length >= min_length:
                palindromes.append(
                    {
                        "start": lo,
                        "end": hi,
                        "length": length,
                        "subsequence": seq[lo : hi + 1],
                    }
                )
    seen = set()
    unique = []
    for p in palindromes:
        key = (p["start"], p["end"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    unique.sort(key=lambda p: -p["length"])
    return DescriptiveResult(
        name="Palindrome detection",
        value=len(unique),
        extra={
            "n": n,
            "min_length": min_length,
            "tolerance": tolerance,
            "palindromes": unique[:20],
            "longest": unique[0]["length"] if unique else 0,
            "coverage": sum(p["length"] for p in unique) / n if unique else 0.0,
        },
    )


paldet = palindrome_detect


def cheatsheet() -> str:
    return 'palindrome_detect({}) -> Palindrome detection in sequences.'
