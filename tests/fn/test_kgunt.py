"""Tests for moirais.fn.kgunt -- Universal kriging trend"""

import numpy as np
import pytest

from moirais.fn.kgunt import uk_trend


class TestUkTrend:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = uk_trend(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = uk_trend(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
