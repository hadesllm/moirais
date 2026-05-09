# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Train BPE merge rules."""

from __future__ import annotations

from collections import Counter

from ._containers import DescriptiveResult


def bpe_train_merges(
    corpus: str,
    vocab_size: int = 256,
) -> DescriptiveResult:
    """Train BPE merge rules from a text corpus.

    Iteratively merges the most frequent adjacent pair of tokens until
    *vocab_size* merges are learned.

    :param corpus: Training text.
    :param vocab_size: Number of merge operations to learn.
    :return: DescriptiveResult with merge rules in ``extra['merges']``.
    """
    tokens = list(corpus)
    merges = []

    for _ in range(vocab_size):
        if len(tokens) < 2:
            break
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1

        if not pairs:
            break

        best = max(pairs, key=pairs.get)
        merges.append(best)

        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                new_tokens.append(best[0] + best[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return DescriptiveResult(
        name="bpe_train_merges",
        value=len(merges),
        extra={"merges": merges, "final_vocab_size": len(set(tokens)), "final_n_tokens": len(tokens)},
    )


def cheatsheet() -> str:
    return "bpe_train_merges(corpus, vocab_size) -> trained BPE merge rules"


bpetm = bpe_train_merges
