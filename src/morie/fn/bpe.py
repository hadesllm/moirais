# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""BPE tokenization."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bpe_encode(
    text: str,
    merges: list[tuple[str, str]],
    vocab: dict[str, int] | None = None,
) -> DescriptiveResult:
    """Encode text using Byte Pair Encoding merge rules.

    :param text: Input text to tokenize.
    :param merges: Ordered list of (a, b) merge pairs.
    :param vocab: Optional token-to-id mapping.
    :return: DescriptiveResult with tokens in ``extra['tokens']``.
    """
    tokens = list(text)

    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    ids = None
    if vocab is not None:
        ids = [vocab.get(t, -1) for t in tokens]

    return DescriptiveResult(
        name="bpe_encode",
        value=len(tokens),
        extra={
            "tokens": tokens,
            "ids": ids,
            "n_tokens": len(tokens),
            "compression_ratio": len(text) / max(len(tokens), 1),
        },
    )


def cheatsheet() -> str:
    return "bpe_encode(text, merges) -> BPE tokenized output"


bpe = bpe_encode
