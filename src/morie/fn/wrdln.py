"""Word length distribution statistics. 'The whole is greater than the sum of its parts. -- Aristotle' -- Chirrut Imwe"""

from __future__ import annotations

import re

import numpy as np

from ._containers import DescriptiveResult


def word_length_stats(text: str) -> DescriptiveResult:
    """
    Compute descriptive statistics of word lengths in *text*.

    :param text: Input text to analyse.
    :type text: str
    :return: DescriptiveResult with mean word length and distribution.
    :raises ValueError: If text is empty or contains no words.

    References
    ----------
    Zipf, G. K. (1949). Human Behavior and the Principle of Least Effort.
    Addison-Wesley.
    """
    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        raise ValueError("Text contains no recognisable words.")

    lengths = np.array([len(w) for w in words])
    unique, counts = np.unique(lengths, return_counts=True)

    return DescriptiveResult(
        name="Word Length Statistics",
        value=float(np.round(np.mean(lengths), 3)),
        extra={
            "n_words": len(words),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0,
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "distribution": dict(zip(unique.tolist(), counts.tolist())),
        },
    )


wrdln = word_length_stats


def cheatsheet() -> str:
    return "word_length_stats({}) -> Word length distribution statistics. 'I am one with the Forc"
