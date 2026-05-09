"""Test rectangular_window (wnrec)."""
import numpy as np
from moirais.fn.wnrec import rectangular_window, wnrec
from moirais.fn._containers import DescriptiveResult


class TestWnrec:
    def test_basic(self):
        result = rectangular_window(16)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "rectangular_window"

    def test_all_ones(self):
        result = rectangular_window(32)
        assert np.all(result.extra["window"] == 1.0)

    def test_alias(self):
        assert wnrec is rectangular_window
