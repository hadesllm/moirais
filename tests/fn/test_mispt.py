"""Tests for missing_pattern."""
import numpy as np, pytest
from moirais.fn.mispt import missing_pattern

class TestMissingPattern:
    def test_basic(self):
        data = np.array([[1, np.nan], [np.nan, 2], [3, 4]])
        r = missing_pattern(data)
        assert r.extra["total_missing"] == 2

    def test_complete(self):
        data = np.ones((10, 3))
        r = missing_pattern(data)
        assert r.extra["complete_rows"] == 10
        assert r.extra["n_patterns"] == 1
