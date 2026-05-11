"""Test box_counting_fd."""
import numpy as np
from morie.fn.boxfd import box_counting_fd, boxfd
from morie.fn._containers import DescriptiveResult


class TestBoxCountingFD:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = box_counting_fd(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_is_float(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = box_counting_fd(x)
        assert isinstance(result.value, float)

    def test_constant_signal(self):
        x = np.ones(100)
        result = box_counting_fd(x)
        assert result.value == 0.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = box_counting_fd(x)
        assert result.name == "box_counting_fd"

    def test_alias(self):
        assert boxfd is box_counting_fd
