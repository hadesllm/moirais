"""Test window_functions."""
import numpy as np
from moirais.fn.winfn import window_functions, winfn
from moirais.fn._containers import DescriptiveResult


class TestWindowFunctions:
    def test_basic(self):
        result = window_functions(256)
        assert isinstance(result, DescriptiveResult)

    def test_hamming_length(self):
        result = window_functions(128, wtype="hamming")
        assert len(result.extra["window"]) == 128

    def test_rectangular(self):
        result = window_functions(64, wtype="rectangular")
        assert np.allclose(result.extra["window"], np.ones(64))

    def test_name(self):
        result = window_functions(256)
        assert result.name == "window_function"

    def test_alias(self):
        assert winfn is window_functions
