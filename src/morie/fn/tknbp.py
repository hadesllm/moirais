# morie.fn — function file (hadesllm/morie)
"""Byte-pair encoding tokenization (Sennrich et al. 2016)."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from ._richresult import RichResult

__all__ = ["bpe_tokenizer"]


def _word_to_symbols(word: str) -> list[str]:
    # Standard BPE: split into characters, mark word boundary with </w>
    return list(word) + ["</w>"]


def _count_pairs(corpus: dict[tuple, int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, freq in corpus.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def _merge_pair(pair: tuple[str, str], corpus: dict[tuple, int]) -> dict[tuple, int]:
    new_corpus: dict[tuple, int] = {}
    a, b = pair
    for symbols, freq in corpus.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(a + b)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_corpus[tuple(new_symbols)] = freq
    return new_corpus


def bpe_tokenizer(x: Iterable[str] | str, num_merges: int = 10):
    """Learn byte-pair-encoding merges from a word corpus.

    Formula: greedily merge the most frequent adjacent symbol pair
    until ``num_merges`` merges have been performed (Sennrich, Haddow
    & Birch 2016, "Neural Machine Translation of Rare Words with
    Subword Units").

    Parameters
    ----------
    x : iterable of str or single str
        Training word list (whitespace-tokenised if a string is passed).
    num_merges : int
        Number of BPE merge operations to learn.

    Returns
    -------
    RichResult with keys: merges (ordered list of pairs), vocab
    (set of subword tokens), n_merges, n_vocab.
    """
    if isinstance(x, str):
        words = x.split()
    else:
        words = list(x)
    if not words:
        return RichResult(payload={"merges": [], "vocab": set(),
                                   "n_merges": 0, "n_vocab": 0,
                                   "method": "BPE"})

    word_freq = Counter(words)
    corpus: dict[tuple, int] = {
        tuple(_word_to_symbols(w)): f for w, f in word_freq.items()
    }
    merges: list[tuple[str, str]] = []
    for _ in range(int(num_merges)):
        pair_counts = _count_pairs(corpus)
        if not pair_counts:
            break
        best, count = pair_counts.most_common(1)[0]
        if count < 1:
            break
        corpus = _merge_pair(best, corpus)
        merges.append(best)
    vocab = {sym for symbols in corpus for sym in symbols}
    return RichResult(
        title="Byte-Pair Encoding (Sennrich 2016)",
        summary_lines=[("Merges learned", len(merges)),
                       ("Vocab size", len(vocab))],
        payload={
            "merges": merges,
            "vocab": vocab,
            "corpus": corpus,
            "n_merges": len(merges),
            "n_vocab": len(vocab),
            "method": "BPE",
        },
    )


def cheatsheet():
    return "tknbp(x, num_merges): byte-pair encoding tokeniser"


# CANONICAL TEST
# >>> r = bpe_tokenizer(["low", "low", "lower", "newest", "newest", "newest"],
# ...                   num_merges=3)
# >>> r["n_merges"]
# 3
