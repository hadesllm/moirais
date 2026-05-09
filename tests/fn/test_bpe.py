"""Test bpe_encode."""
from moirais.fn.bpe import bpe_encode, bpe
from moirais.fn._containers import DescriptiveResult


class TestBpeEncode:
    def test_basic(self):
        merges = [("a", "b")]
        result = bpe_encode("abc", merges)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bpe_encode"

    def test_merge(self):
        merges = [("a", "b")]
        result = bpe_encode("ab", merges)
        assert result.extra["tokens"] == ["ab"]

    def test_no_merge(self):
        merges = [("x", "y")]
        result = bpe_encode("ab", merges)
        assert result.extra["tokens"] == ["a", "b"]

    def test_with_vocab(self):
        merges = [("a", "b")]
        vocab = {"ab": 0, "c": 1}
        result = bpe_encode("abc", merges, vocab=vocab)
        assert result.extra["ids"] == [0, 1]

    def test_alias(self):
        assert bpe is bpe_encode
