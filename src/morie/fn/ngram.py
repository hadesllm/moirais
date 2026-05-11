# morie.fn — function file (hadesllm/morie)
"""Without music, life would be a mistake. — Friedrich Nietzsche"""

from __future__ import annotations

from collections import Counter

from ._containers import DescriptiveResult


def ngram_freq(
    text: str,
    n: int = 2,
    top_k: int = 50,
) -> DescriptiveResult:
    """Compute n-gram frequencies from text.

    Tokenizes on whitespace, builds character or word n-grams, and
    returns ranked frequency counts.

    Parameters
    ----------
    text : str
        Input text.
    n : int
        Size of the n-gram (2=bigrams, 3=trigrams, etc.).
    top_k : int
        Number of top n-grams to return.

    Returns
    -------
    DescriptiveResult
        name='N-gram Frequency', value=top_k count,
        extra has 'ngrams' (list of (ngram, count) tuples),
        'total_ngrams', 'unique_ngrams'.

    References
    ----------
    Jurafsky, D. & Martin, J.H. (2023). *Speech and Language
    Processing* (3rd ed.). Ch. 3: N-gram Language Models.
    https://web.stanford.edu/~jurafsky/slp3/
    """
    if not text or not text.strip():
        return DescriptiveResult(
            name="N-gram Frequency",
            value=0,
            extra={"ngrams": [], "total_ngrams": 0, "unique_ngrams": 0},
        )

    tokens = text.lower().split()
    if len(tokens) < n:
        return DescriptiveResult(
            name="N-gram Frequency",
            value=0,
            extra={"ngrams": [], "total_ngrams": 0, "unique_ngrams": 0},
        )

    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    top = counts.most_common(top_k)

    top_readable = [(" ".join(g), c) for g, c in top]

    return DescriptiveResult(
        name="N-gram Frequency",
        value=len(top),
        extra={
            "ngrams": top_readable,
            "total_ngrams": len(grams),
            "unique_ngrams": len(counts),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "ngram_freq({}) -> N-gram frequency analysis. 'I find your lack of faith distur"
