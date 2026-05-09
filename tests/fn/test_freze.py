"""Tests for moirais.fn.freze -- Freeze-thaw degradation."""

from moirais.fn.freze import freeze_thaw, freze
from moirais.fn._containers import DescriptiveResult


class TestFreze:
    def test_alias(self):
        assert freze is freeze_thaw

    def test_degrades(self):
        result = freeze_thaw(100, initial_strength=100, degradation_rate=0.05, threshold=50, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.value is not None
        assert result.value < 100

    def test_no_failure(self):
        result = freeze_thaw(10, initial_strength=100, degradation_rate=0.001, threshold=50, seed=42)
        assert result.value is None
