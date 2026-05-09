"""Test higuchi_fd."""
import numpy as np
from moirais.fn.higfd import higuchi_fd, higfd
from moirais.fn._containers import DescriptiveResult


class TestHiguchiFD:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(512)
        result = higuchi_fd(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_is_float(self):
        x = np.random.default_rng(42).standard_normal(512)
        result = higuchi_fd(x)
        assert isinstance(result.value, float)

    def test_white_noise_fd(self):
        x = np.random.default_rng(42).standard_normal(1024)
        result = higuchi_fd(x, kmax=10)
        assert 0.5 < result.value < 3.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(512)
        result = higuchi_fd(x)
        assert result.name == "higuchi_fd"

    def test_alias(self):
        assert higfd is higuchi_fd
