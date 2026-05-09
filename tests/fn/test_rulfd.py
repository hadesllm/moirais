"""Test ruler_fd."""
import numpy as np
from moirais.fn.rulfd import ruler_fd, rulfd
from moirais.fn._containers import DescriptiveResult


class TestRulerFD:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = ruler_fd(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_is_float(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = ruler_fd(x)
        assert isinstance(result.value, float)

    def test_smooth_signal(self):
        t = np.linspace(0, 2 * np.pi, 256)
        x = np.sin(t)
        result = ruler_fd(x)
        assert result.value > 0.5

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = ruler_fd(x)
        assert result.name == "ruler_fd"

    def test_alias(self):
        assert rulfd is ruler_fd
