"""Test bpe_train_merges."""
from morie.fn.bpetm import bpe_train_merges, bpetm
from morie.fn._containers import DescriptiveResult


class TestBpeTrainMerges:
    def test_basic(self):
        result = bpe_train_merges("aaabbb", vocab_size=3)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bpe_train_merges"

    def test_learns_merges(self):
        result = bpe_train_merges("ababab", vocab_size=2)
        assert len(result.extra["merges"]) > 0

    def test_compression(self):
        result = bpe_train_merges("aaaa", vocab_size=5)
        assert result.extra["final_n_tokens"] <= 4

    def test_alias(self):
        assert bpetm is bpe_train_merges
