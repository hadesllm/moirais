"""Test lr_finder."""
import numpy as np
from moirais.fn.lrinf import lr_finder, lrinf
from moirais.fn._containers import DescriptiveResult


class TestLrFinder:
    def test_basic(self):
        losses = [2.0, 1.5, 1.0, 0.8, 0.9, 1.5, 3.0]
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        result = lr_finder(losses, lrs)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "lr_finder"

    def test_finds_descent(self):
        losses = [3.0, 2.5, 2.0, 1.0, 0.5, 1.0, 5.0]
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        result = lr_finder(losses, lrs)
        assert result.value > 0

    def test_mismatched_length(self):
        import pytest
        with pytest.raises(ValueError):
            lr_finder([1.0, 2.0], [1e-3])

    def test_alias(self):
        assert lrinf is lr_finder
