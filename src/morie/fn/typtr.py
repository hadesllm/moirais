"""You have power over your mind -- not outside events. -- Marcus Aurelius"""

from __future__ import annotations

import re

import numpy as np

from ._containers import DescriptiveResult


def type_token_ratio(text: str) -> DescriptiveResult:
    """
    Compute the Type-Token Ratio (TTR) as a measure of lexical diversity.

    .. math::

        TTR = \\frac{|\\text{types}|}{|\\text{tokens}|}

    where types are unique word forms and tokens are total words.

    :param text: Input text to analyse.
    :type text: str
    :return: DescriptiveResult with TTR value and counts.
    :raises ValueError: If text is empty or contains no words.

    References
    ----------
    Templin, M. C. (1957). Certain language skills in children.
    University of Minnesota Press.
    """
    tokens = [w.lower() for w in re.findall(r"[a-zA-Z]+", text)]
    if not tokens:
        raise ValueError("Text contains no recognisable words.")

    types = set(tokens)
    ttr = len(types) / len(tokens)

    return DescriptiveResult(
        name="Type-Token Ratio",
        value=float(np.round(ttr, 4)),
        extra={"n_types": len(types), "n_tokens": len(tokens)},
    )


typtr = type_token_ratio


def cheatsheet() -> str:
    return "type_token_ratio({}) -> Type-token ratio (lexical diversity). 'Never tell me the odd"
