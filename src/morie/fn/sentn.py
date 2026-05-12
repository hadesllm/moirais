# morie.fn -- function file (hadesllm/morie)
"""Lexicon-based sentiment analysis. 'There is always a bigger fish.' -- Qui-Gon Jinn"""

from __future__ import annotations

from ._containers import DescriptiveResult

_DEFAULT_POS = frozenset(
    {
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "positive",
        "happy",
        "love",
        "best",
        "better",
        "nice",
        "fine",
        "beautiful",
        "brilliant",
        "outstanding",
        "superb",
        "perfect",
        "pleasant",
        "helpful",
        "effective",
        "strong",
        "success",
        "benefit",
        "improve",
        "advantage",
        "well",
        "healthy",
        "safe",
        "protective",
    }
)

_DEFAULT_NEG = frozenset(
    {
        "bad",
        "terrible",
        "awful",
        "horrible",
        "worst",
        "poor",
        "negative",
        "hate",
        "ugly",
        "dreadful",
        "disappointing",
        "failure",
        "weak",
        "harm",
        "danger",
        "risk",
        "toxic",
        "disease",
        "death",
        "pain",
        "worse",
        "problem",
        "difficult",
        "unfortunate",
        "sad",
        "angry",
        "fear",
        "threat",
        "damage",
        "decline",
        "loss",
        "adverse",
    }
)


def sentiment_lexicon(
    text: str,
    pos_words: list[str] | set[str] | None = None,
    neg_words: list[str] | set[str] | None = None,
) -> DescriptiveResult:
    """Lexicon-based sentiment scoring.

    Counts positive and negative word occurrences and computes a
    normalized sentiment score in [-1, 1].

    Parameters
    ----------
    text : str
        Input text to analyze.
    pos_words : list or set or None
        Custom positive lexicon. Uses built-in if None.
    neg_words : list or set or None
        Custom negative lexicon. Uses built-in if None.

    Returns
    -------
    DescriptiveResult
        name='Sentiment', value=sentiment score in [-1, 1],
        extra has 'pos_count', 'neg_count', 'total_words',
        'pos_matches', 'neg_matches'.

    References
    ----------
    Liu, B. (2012). Sentiment Analysis and Opinion Mining.
    *Synthesis Lectures on Human Language Technologies*, 5(1), 1-167.
    doi:10.2200/S00416ED1V01Y201204HLT016
    """
    if not text or not text.strip():
        return DescriptiveResult(
            name="Sentiment",
            value=0.0,
            extra={"pos_count": 0, "neg_count": 0, "total_words": 0, "pos_matches": [], "neg_matches": []},
        )

    pos_set = frozenset(pos_words) if pos_words is not None else _DEFAULT_POS
    neg_set = frozenset(neg_words) if neg_words is not None else _DEFAULT_NEG

    tokens = ["".join(c for c in w if c.isalnum()) for w in text.lower().split()]
    tokens = [t for t in tokens if t]

    pos_matches = [t for t in tokens if t in pos_set]
    neg_matches = [t for t in tokens if t in neg_set]

    pos_count = len(pos_matches)
    neg_count = len(neg_matches)
    total = pos_count + neg_count

    score = (pos_count - neg_count) / total if total > 0 else 0.0

    return DescriptiveResult(
        name="Sentiment",
        value=float(score),
        extra={
            "pos_count": pos_count,
            "neg_count": neg_count,
            "total_words": len(tokens),
            "pos_matches": list(set(pos_matches)),
            "neg_matches": list(set(neg_matches)),
        },
    )


def cheatsheet() -> str:
    return "sentiment_lexicon({}) -> Lexicon-based sentiment analysis. 'There is always a bigger "
