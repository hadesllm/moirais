# morie.fn -- function file (hadesllm/morie)
"""Flesch-Kincaid readability grade level. 'Your focus determines your reality.' -- Qui-Gon Jinn"""

from __future__ import annotations

import re

import numpy as np

from ._containers import DescriptiveResult


def flesch_kincaid(text: str) -> DescriptiveResult:
    """
    Compute the Flesch-Kincaid Grade Level readability score.

    .. math::

        FK = 0.39 \\frac{\\text{words}}{\\text{sentences}}
             + 11.8 \\frac{\\text{syllables}}{\\text{words}} - 15.59

    :param text: Input text to analyse.
    :type text: str
    :return: DescriptiveResult with grade level and component counts.
    :raises ValueError: If text is empty or contains no words.

    References
    ----------
    Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975).
    Derivation of new readability formulas for Navy enlisted personnel.
    Research Branch Report 8-75, Naval Technical Training Command.
    """
    if not text or not text.strip():
        raise ValueError("Text must be non-empty.")

    sentences = max(len(re.split(r"[.!?]+", text.strip())) - 1, 1)
    words_list = re.findall(r"[a-zA-Z]+", text)
    n_words = len(words_list)
    if n_words == 0:
        raise ValueError("Text contains no recognisable words.")

    def _count_syllables(word: str) -> int:
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub(r"e$", "", word)
        vowels = re.findall(r"[aeiouy]+", word)
        return max(len(vowels), 1)

    n_syllables = sum(_count_syllables(w) for w in words_list)

    grade = 0.39 * (n_words / sentences) + 11.8 * (n_syllables / n_words) - 15.59

    return DescriptiveResult(
        name="Flesch-Kincaid Grade Level",
        value=float(np.round(grade, 2)),
        extra={"words": n_words, "sentences": sentences, "syllables": n_syllables},
    )


fkrad = flesch_kincaid


def cheatsheet() -> str:
    return "flesch_kincaid({}) -> Flesch-Kincaid readability grade level. 'Your focus determin"
