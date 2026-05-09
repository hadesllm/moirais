"""Tests for moirais.fn.zstrd -- Temporal trend estimation"""

import numpy as np
import pytest

from moirais.fn.zstrd import trend_temporal


class TestTrendTemporal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = trend_temporal(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = trend_temporal(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
