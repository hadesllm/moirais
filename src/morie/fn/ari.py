# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""It is the mark of an educated mind to entertain a thought without accepting it. -- Aristotle"""

from __future__ import annotations

import re

import numpy as np

from ._containers import DescriptiveResult


def automated_readability(text: str) -> DescriptiveResult:
    """
    Compute the Automated Readability Index (ARI).

    .. math::

        ARI = 4.71 \\frac{\\text{characters}}{\\text{words}}
              + 0.5 \\frac{\\text{words}}{\\text{sentences}} - 21.43

    :param text: Input text to analyse.
    :type text: str
    :return: DescriptiveResult with ARI score and component counts.
    :raises ValueError: If text is empty or contains no words.

    References
    ----------
    Senter, R. J., & Smith, E. A. (1967). Automated Readability Index.
    AMRL-TR-66-220. Wright-Patterson Air Force Base.
    """
    if not text or not text.strip():
        raise ValueError("Text must be non-empty.")

    sentences = max(len(re.split(r"[.!?]+", text.strip())) - 1, 1)
    words_list = re.findall(r"[a-zA-Z]+", text)
    n_words = len(words_list)
    if n_words == 0:
        raise ValueError("Text contains no recognisable words.")

    n_chars = sum(len(w) for w in words_list)
    score = 4.71 * (n_chars / n_words) + 0.5 * (n_words / sentences) - 21.43

    return DescriptiveResult(
        name="Automated Readability Index",
        value=float(np.round(score, 2)),
        extra={"characters": n_chars, "words": n_words, "sentences": sentences},
    )


ari = automated_readability


def cheatsheet() -> str:
    return "automated_readability({}) -> Automated Readability Index. 'In my experience there is no s"
