"""Test data_loader_stats."""
from moirais.fn.dtldr import data_loader_stats, dtldr
from moirais.fn._containers import DescriptiveResult


class TestDataLoaderStats:
    def test_basic(self):
        result = data_loader_stats(1000000, batch_size=32, seq_len=512)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "data_loader_stats"

    def test_iters(self):
        result = data_loader_stats(10000, batch_size=10, seq_len=100)
        assert result.extra["iters_per_epoch"] == 10

    def test_multi_epoch(self):
        result = data_loader_stats(10000, batch_size=10, seq_len=100, n_epochs=3)
        assert result.extra["total_steps"] == 30

    def test_alias(self):
        assert dtldr is data_loader_stats
