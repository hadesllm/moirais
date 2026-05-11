"""Test split_signal (sgspl)."""
import numpy as np
from morie.fn.sgspl import split_signal, sgspl
from morie.fn._containers import DescriptiveResult


class TestSplitSignal:
    def test_basic(self):
        x = np.arange(100, dtype=float)
        result = split_signal(x, segment_len=25)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 4.0

    def test_segments_shape(self):
        x = np.arange(100, dtype=float)
        result = split_signal(x, segment_len=30)
        assert result.extra["segments"].shape == (3, 30)
        assert len(result.extra["remainder"]) == 10

    def test_alias(self):
        assert sgspl is split_signal
