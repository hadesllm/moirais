# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bits per character (BPC)."""

import math

from ._containers import ESRes


def bits_per_char(text: str, **kwargs) -> ESRes:
    """
    Compute bits per character using character-level entropy.

    Estimates the empirical entropy of a text string by computing
    character frequency distribution and Shannon entropy.

    :param text: Input string.
    :return: ESRes with bits per character.

    References
    ----------
    Shannon CE (1951). Prediction and entropy of printed English.
    Bell System Technical Journal, 30(1), 50-64.
    """
    if not text:
        raise ValueError("Text must be non-empty.")
    n = len(text)
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    bpc = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            bpc -= p * math.log2(p)
    return ESRes(
        measure="bits_per_char",
        estimate=bpc,
        n=n,
        extra={"n_unique_chars": len(freq), "text_length": n},
    )


bpenc = bits_per_char


def cheatsheet() -> str:
    return "bits_per_char(text) -> Bits per character."
