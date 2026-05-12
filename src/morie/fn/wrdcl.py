"""Patience is bitter, but its fruit is sweet. -- Aristotle"""

from __future__ import annotations

from collections import Counter

from ._containers import DescriptiveResult

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
    }
)


def word_cloud_data(
    text: str,
    top_k: int = 50,
    remove_stopwords: bool = True,
    min_length: int = 2,
) -> DescriptiveResult:
    """Compute word frequencies suitable for word cloud visualization.

    Parameters
    ----------
    text : str
        Input text corpus.
    top_k : int
        Number of top words to return.
    remove_stopwords : bool
        Whether to remove common English stop words.
    min_length : int
        Minimum word length to include.

    Returns
    -------
    DescriptiveResult
        name='Word Cloud Data', value=top_k count,
        extra has 'words' (list of (word, freq) tuples),
        'total_words', 'unique_words'.

    References
    ----------
    Heimerl, F. et al. (2014). Word Cloud Explorer: Text Analytics
    Based on Word Clouds. *47th Hawaii International Conference on
    System Sciences*, 1833-1842. doi:10.1109/HICSS.2014.231
    """
    if not text or not text.strip():
        return DescriptiveResult(
            name="Word Cloud Data",
            value=0,
            extra={"words": [], "total_words": 0, "unique_words": 0},
        )

    tokens = []
    for w in text.lower().split():
        cleaned = "".join(c for c in w if c.isalnum())
        if len(cleaned) >= min_length:
            if not remove_stopwords or cleaned not in _STOP_WORDS:
                tokens.append(cleaned)

    counts = Counter(tokens)
    top = counts.most_common(top_k)

    max_count = top[0][1] if top else 1
    normalized = [(w, c, round(c / max_count, 4)) for w, c in top]

    return DescriptiveResult(
        name="Word Cloud Data",
        value=len(top),
        extra={
            "words": top,
            "normalized": normalized,
            "total_words": len(tokens),
            "unique_words": len(counts),
        },
    )


def cheatsheet() -> str:
    return "word_cloud_data({}) -> Word frequency data for cloud visualization. 'Never tell me "
