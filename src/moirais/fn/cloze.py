# moirais.fn — function file (hadesllm/moirais)
"""Cloze test scoring. 'Truly wonderful the mind of a child is.'"""

from __future__ import annotations

import re

import numpy as np

from ._containers import DescriptiveResult


def cloze_score(text: str, n_deletions: int = 20, seed: int = 42) -> DescriptiveResult:
    """
    Generate a cloze test from *text* by deleting *n_deletions* words.

    Returns the deletion positions and the expected answers so that a
    respondent's fill-in accuracy can be scored as
    ``correct / n_deletions``.

    :param text: Source text from which words are deleted.
    :type text: str
    :param n_deletions: Number of words to blank out. Default 20.
    :type n_deletions: int
    :param seed: RNG seed for reproducibility. Default 42.
    :type seed: int
    :return: DescriptiveResult with deletion count and answer key.
    :raises ValueError: If text has fewer words than *n_deletions*.

    References
    ----------
    Taylor, W. L. (1953). Cloze procedure: a new tool for measuring
    readability. Journalism Quarterly, 30(4), 415--433.
    """
    words = re.findall(r"\S+", text)
    if len(words) < n_deletions:
        raise ValueError(f"Text has {len(words)} words but {n_deletions} deletions requested.")

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(words), size=n_deletions, replace=False))
    answers = {int(i): words[i] for i in indices}

    return DescriptiveResult(
        name="Cloze Test",
        value=n_deletions,
        extra={
            "total_words": len(words),
            "deletion_indices": indices.tolist(),
            "answer_key": answers,
        },
    )


cloze = cloze_score


def cheatsheet() -> str:
    return "cloze_score({}) -> Cloze test scoring. 'Truly wonderful the mind of a child is."
