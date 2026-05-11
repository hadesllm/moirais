"""Test sample_bound (smpbd)."""
import pytest

from morie.fn.smpbd import sample_bound, smpbd
from morie.fn._containers import DescriptiveResult


class TestSampleBound:
    def test_default(self):
        result = sample_bound()
        assert isinstance(result, DescriptiveResult)
        assert result.name == "sample_bound"
        assert result.value == 385.0

    def test_narrow_margin(self):
        result = sample_bound(margin=0.01)
        assert result.value > 385

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            sample_bound(confidence=1.5)

    def test_alias(self):
        assert smpbd is sample_bound
