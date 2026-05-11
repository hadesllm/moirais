"""Tests for morie.fn.nmdwt -- DW-NOMINATE trend analysis"""

import numpy as np
import pytest

from morie.fn.nmdwt import dwnominate_trend


class TestDwnominateTrend:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dwnominate_trend(data)
        assert result.value is not None

    def test_output_type(self):
        result = dwnominate_trend(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
