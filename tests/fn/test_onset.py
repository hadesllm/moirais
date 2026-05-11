"""Test onset_detect_fn."""
import numpy as np
from morie.fn.onset import onset_detect_fn, alias
from morie.fn._containers import DescriptiveResult


class TestOnsetDetectFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = onset_detect_fn(x, fs=1000.0)
        assert isinstance(result, DescriptiveResult)

    def test_value(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = onset_detect_fn(x, fs=1000.0)
        assert isinstance(result.value, int)
        assert result.value >= 0

    def test_onsets_in_extra(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = onset_detect_fn(x, fs=1000.0)
        assert "onsets" in result.extra

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = onset_detect_fn(x, fs=1000.0)
        assert result.name == "onset_detect"

    def test_alias(self):
        assert alias is onset_detect_fn
