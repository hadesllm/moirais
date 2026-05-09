"""Test bartlett_window (wnbrt)."""
import numpy as np
from moirais.fn.wnbrt import bartlett_window, wnbrt
from moirais.fn._containers import DescriptiveResult


class TestWnbrt:
    def test_basic(self):
        result = bartlett_window(16)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bartlett_window"

    def test_zero_endpoints(self):
        result = bartlett_window(32)
        w = result.extra["window"]
        assert np.isclose(w[0], 0.0, atol=1e-10)
        assert np.isclose(w[-1], 0.0, atol=1e-10)

    def test_alias(self):
        assert wnbrt is bartlett_window
